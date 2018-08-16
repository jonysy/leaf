#[macro_use]
extern crate log;
#[macro_use]
extern crate serde_derive;

extern crate csv;
extern crate docopt;
extern crate env_logger;
extern crate hyper;
extern crate hyper_tls;
extern crate leaf;
extern crate parenchyma;
extern crate parenchyma_ml;
extern crate tokio;

use csv::Reader;
use docopt::Docopt;
use hyper::{Body, Client};
use hyper::client::HttpConnector;
use hyper::rt::{Future, Stream};
use hyper_tls::HttpsConnector;
use leaf::layers::*;
use leaf::solvers::*;
use leaf::typedefs::ArcLockTensor;
use parenchyma::frameworks::Native;
use parenchyma::prelude::{Backend, SharedTensor};
use parenchyma_ml::Package as MachLrnPackage;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, RwLock};
use tokio::runtime::Runtime;

type NativeMachLrnPackage = Native<MachLrnPackage>;

const USAGE: &'static str = "
Leaf Examples

Usage:
    leaf-examples load-dataset <dataset-name>
    leaf-examples mnist <model-name> [--batch-size <batch-size>] [--learning-rate <learning-rate>] [--momentum <momentum>]
    leaf-examples (-h | --help)
    leaf-examples --version

Options:
    <model-name>            Which MNIST model to use. Valid values: [linear, mlp]

    -h --help               Show this screen.
    --version               Show version.
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_dataset_name: Option<String>,
    arg_model_name: Option<String>,
    arg_batch_size: Option<usize>,
    arg_learning_rate: Option<f32>,
    arg_momentum: Option<f32>,
    cmd_load_dataset: bool,
    cmd_mnist: bool,
}

fn main() {
    env_logger::init();

    let args: Args = 
        Docopt::new(USAGE)
            .and_then(|d| d.deserialize())
                .unwrap_or_else(|e| e.exit());

    if args.cmd_load_dataset {
        match args.arg_dataset_name.unwrap().as_ref() {
            "mnist" => {
                let ref mut rt = Runtime::new().unwrap();
                let ref client = Client::builder().build(HttpsConnector::new(4).unwrap());
                ["mnist_test.csv", "mnist_train.csv"]
                    .iter()
                        .for_each(|path| fetch(rt, client, path));
                println!("MNIST dataset downloaded");
            }
            _ => {
                println!("Failed to download MNIST dataset!");
            }
        }
    } else if args.cmd_mnist {
        run_mnist(
            args.arg_model_name.unwrap_or("none".to_owned()), 
            args.arg_batch_size.unwrap_or(1), 
            args.arg_learning_rate.unwrap_or(0.001f32), 
            args.arg_momentum.unwrap_or(0f32)
        );
    }
}


fn fetch(runtime: &mut Runtime, client: &Client<HttpsConnector<HttpConnector>, Body>, path: &str) {
    let file_path = Path::new("assets").join(path);

    if !file_path.exists() {
        println!("Downloading `{}`..", path);
        let mut file = File::create(file_path).unwrap();
        let work = 
            client
                .get( format!("https://pjreddie.com/media/files/{}", path).parse().unwrap() )
                .and_then(move |res| {
                    // The body is a stream, and for_each returns a new Future
                    // when the stream is finished, and calls the closure on
                    // each chunk of the body...
                    res.into_body().for_each(move |chunk| {
                        file.write_all(&chunk)
                            .map_err(|e| panic!("expects file is open, error={}", e))
                    })
                })
                // If all good, just tell the user...
                .map(|_| {
                    println!("Done.");
                })
                // If there was an error, let the user know...
                .map_err(|err| {
                    eprintln!("Error {}", err);
                });

        runtime.block_on(work).unwrap()
    } else {
        println!("`{}` already downloaded..", path);
    }
}

fn run_mnist(model_name: String, batch_size: usize, learning_rate: f32, momentum: f32) {
    let mut reader = Reader::from_path("assets/mnist_train.csv").unwrap();
    let mut decoded_images = 
        reader.deserialize().map(|row| match row {
            Ok(value) => {
                let row_vec: Box<Vec<u8>> = Box::new(value);
                let label = row_vec[0];
                let mut pixels = vec![0u8; 784];
                for (place, element) in pixels.iter_mut().zip(row_vec.iter().skip(1)) {
                    *place = *element;
                }

                // TODO: reintroduce pre-processing framework
                // let img = Image::from_luma_pixels(28, 28, pixels);
                // match img {
                //     Ok(in_img) => {
                //         println!("({}): {:?}", label, in_img.transform(vec![28, 28]));
                //     },
                //     _ => unimplemented!()
                // }

                (label, pixels)
            }

            _ => {
                println!("no value");
                panic!();
            }
        });

// -------------------------------------------------------------------------------------------------

    let mut net_cfg = SequentialConfig::default();
    net_cfg.add_input("data", &[batch_size, 28, 28]);
    net_cfg.force_backward = true;

    match model_name.as_ref() {
        "conv" => {
            unimplemented!()
        }

        "mlp" => {
            net_cfg.add_layer(LayerConfig::new("reshape", 
                LayerType::Reshape(ReshapeConfig::of_shape(&[batch_size, 784]))));
            net_cfg.add_layer(LayerConfig::new("linear1", 
                LayerType::Linear(LinearConfig { output_size: 1568 })));
            net_cfg.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            net_cfg.add_layer(LayerConfig::new("linear2", 
                LayerType::Linear(LinearConfig { output_size: 10 })));
        }

        "linear" => {
            net_cfg.add_layer(LayerConfig::new("linear", 
                LayerType::Linear(LinearConfig { output_size: 10 })));
        }

        _ => {
            panic!("Unknown model. Try one of [linear, mlp, conv]")
        }
    }

    net_cfg.add_layer(LayerConfig::new("log_softmax", LayerType::LogSoftmax));

    let mut classifier_cfg = SequentialConfig::default();
    classifier_cfg.add_input("network_out", &[batch_size, 10]);
    classifier_cfg.add_input("label", &[batch_size, 1]);
    // set up nll loss
    let nll_layer_cfg = NegativeLogLikelihoodConfig { num_classes: 10 };
    let nll_cfg = LayerConfig::new("nll", LayerType::NegativeLogLikelihood(nll_layer_cfg));
    classifier_cfg.add_layer(nll_cfg);

    // set up backends
    let backend = ::std::rc::Rc::new(
        Backend::new::<NativeMachLrnPackage>().unwrap());

    // set up solver
    let solver_cfg = SolverConfig {
        minibatch_size: batch_size,
        base_lr: learning_rate,
        momentum: momentum,
        network: LayerConfig::new("network", net_cfg),
        objective: LayerConfig::new("classifier", classifier_cfg),
        .. SolverConfig::default()
    };
    let mut solver = Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    // set up confusion matrix
    let mut confusion = ::leaf::solvers::ConfusionMatrix::new(10);
    confusion.set_capacity(Some(1000));

// -------------------------------------------------------------------------------------------------

    let in_lock: ArcLockTensor = Arc::new(RwLock::new(SharedTensor::from([batch_size, 1, 28, 28])));
    let label_lock: ArcLockTensor = Arc::new(RwLock::new(SharedTensor::from([batch_size, 1])));

    for _ in 0..(60000 / batch_size) {
        let mut targets: Vec<usize> = 
            decoded_images
                .by_ref()
                .take(batch_size)
                .enumerate().map(|(batch_n, (label_val, input))| {
                    let mut in_tensor = in_lock.write().unwrap();
                    let mut label_tensor = label_lock.write().unwrap();
                    in_tensor.write_batch_sample(&input, batch_n);
                    label_tensor.write_batch_sample(&[label_val], batch_n);
                    label_val as usize
                })
                .collect();

        // train the network!
        let infered_out = solver.train_minibatch(in_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = confusion.get_predictions(&mut infered);
        confusion.add_samples(&predictions, &targets);
        println!("Last sample: {} | Accuracy {}",
            confusion.samples().iter().last().unwrap(), confusion.accuracy());
    }
}