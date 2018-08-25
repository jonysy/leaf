# leaf

[![License](https://img.shields.io/crates/l/leaf.svg)](LICENSE)

## Introduction

Leaf is a open machine learning framework for hackers to build classical, deep
or hybrid machine learning applications. It was inspired by the brilliant people
behind TensorFlow, Torch, Caffe, Rust and numerous research papers and brings
modularity, performance and portability to deep learning.

Leaf has one of the simplest APIs, is lean and tries to introduce minimal
technical debt to your stack.

See the _Leaf - Machine Learning for Hackers_ book for more.

Thanks to Leaf's architecture and Rust, it is already
one of the fastest machine intelligence frameworks available.

Leaf is portable. Run it on CPUs, GPUs, and FPGAs, on machines with an OS, or on
machines without one. Run it with OpenCL or CUDA. Credit goes to
~~Collenchyma~~ Parenchyma and Rust.

~~Leaf is part of the Autumn Machine Intelligence Platform, which is
working on making AI algorithms 100x more computational efficient.~~

We see Leaf as the core of constructing high-performance machine intelligence
applications. Leaf's design makes it easy to publish independent modules to make
e.g. deep reinforcement learning, visualization and monitoring, network
distribution, automated preprocessing or scaleable production
deployment easily accessible for everyone.

Disclaimer: Leaf is currently in an early stage of development. If you are experiencing any bugs 
with features that have been implemented, feel free to create a issue.

## Getting Started

### Documentation

To learn how to build classical, deep or hybrid machine learning applications with Leaf, check out 
the _Leaf - Machine Learning for Hackers_ book.

For additional information see the Rust API documentation.

Or start by running the **Leaf examples**.

We are providing a Leaf examples repository, where we and
others publish executable machine learning models build with Leaf. It features
a CLI for easy usage and has a detailed guide in the project README.md.

Leaf comes with an examples directory as well, which features popular neural
networks (e.g. Alexnet, Overfeat, VGG). To run them on your machine, just follow
the install guide, clone this repoistory and then run

```bash
# The examples currently require CUDA or OpenCL support.
cargo run --release --no-default-features --features cuda --example benchmarks alexnet
```

### Installation

Leaf is built in Rust. If you are new to Rust you can install Rust. We also recommend taking a 
look at the official _Rust - Getting Started Guide_.

To start building a machine learning application (Rust only for now but wrappers are welcome) and 
you are using Cargo, just add Leaf to your `Cargo.toml`:

```toml
[dependencies]
leaf = "<version>"
```

### Contributing

If you want to start hacking on Leaf (e.g. adding a new `Layer`) you should start with forking 
and cloning the repository.

We have more instructions to help you get started in the CONTRIBUTING.md.

We also has a near real-time collaboration culture, which happens here on Github and on 
the Leaf Gitter channel.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in 
the work by you, as defined in the Apache-2.0 license, shall be dual licensed as below, without 
any additional terms or conditions.

## Ecosystem / Extensions

We designed Leaf and other crates to be as modular and extensible as possible. More helpful 
crates you can use with Leaf:

- ~~Cuticula~~ Parenchyma-tr: Preprocessing Framework for Machine Learning
- ~~Collenchyma~~ Parenchyma: Portable, HPC-Framework on any hardware with CUDA, OpenCL, Rust

## Support / Contact

- With a bit of luck, you can find us online on the #rust-machine-learning IRC at irc.mozilla.org,
- but we are always approachable on Gitter
- For bugs and feature request, you can create a Github issue
- For more private matters, send us email straight to our inbox

## Changelog

You can find the release history at the CHANGELOG.md. We are using Clog, the Rust tool for 
auto-generating CHANGELOG files.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
