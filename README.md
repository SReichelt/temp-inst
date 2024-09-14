# temp-inst crate

This crate provides safe lifetime-erased representations for objects with lifetime parameters.
Safety is achieved by making the lifetime-erased objects accessible only via short-lived
references.

The main use case is to convert multiple (shared or mutable) references into a single reference
to a lifetime-erased object, which can then be passed to an API that only accepts a single
reference.

## Example

```rust
// We want to implement this example trait for a specific type `Bar`, in order to call
// `run_twice` below.
pub trait Foo {
    type Arg;

    fn run(arg: &mut Self::Arg);
}

pub fn run_twice<F: Foo>(arg: &mut F::Arg) {
    F::run(arg);
    F::run(arg);
}

struct Bar;

impl Foo for Bar {
    // We actually want to use _two_ mutable references as the argument type. However,
    // the associated type `Arg` does not have any lifetime parameter. If we can add a
    // lifetime parameter `'a` to `Bar`, then `type Arg = (&'a mut i32, &'a mut i32)`
    // will work. If we can't or don't want to do that, a pair of `TempRefMut` will do
    // the trick.
    type Arg = (TempRefMut<i32>, TempRefMut<i32>);

    fn run(arg: &mut Self::Arg) {
        // The mutable `TempRefMut` references can be dereferenced to obtain the mutable
        // references that we passed to `call_with` below.
        let (a_ref, b_ref) = arg;
        **a_ref += **b_ref;
        **b_ref += 1;
    }
}

let mut a = 42;
let mut b = 23;

// Now we can convert the pair `(&mut a, &mut b)` to `&mut Foo::Arg`, and pass that to
// `run_twice`.
TempInstMut::call_with((&mut a, &mut b), run_twice::<Bar>);

assert_eq!(a, 42 + 23 + 1 + 23);
assert_eq!(b, 23 + 1 + 1);
```

See the [source code](temp-inst/src/lib.rs) for more info.

## Similar crates

[dyn-context](https://crates.io/crates/dyn-context) is another solution for the same problem. The
main difference is that it focuses on user-defined types, whereas temp-inst defines a trait that is
already implemented for standard types but must be derived or implemented manually for custom types.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT License
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
