extern crate capnpc;

fn main() {
    ::capnpc::compile("capnp", &["capnp/cerealization_protocol.capnp"]).unwrap();
}
