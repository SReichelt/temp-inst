# temp-inst crate

[`TempInst`](temp-inst/src/lib.rs) is a safe lifetime-erased wrapper for a Rust object (struct,
enum, tuple, ...) with lifetime parameters. It is intended for use with APIs that require a single
(shared or mutable) reference to an object that cannot depend on lifetimes (except for the lifetime
of that single reference).

For example, `TempInst` can be used to contract a tuple of references into a single reference
and then later back into a tuple of references. For the full list of types that can be used
with `TempInst`, see the implementations of the `TempRepr` trait.

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
    // will work. If we can't or don't want to do that, an equivalent `TempInst` will
    // do the trick.
    type Arg = TempInst<(TempRefMut<i32>, TempRefMut<i32>)>;

    fn run(arg: &mut Self::Arg) {
        // From a mutable `TempInst` reference, we can extract the mutable references
        // that we originally constructed it from.
        let (a_ref, b_ref) = arg.get_mut();
        *a_ref += *b_ref;
        *b_ref += 1;
    }
}

let mut a = 42;
let mut b = 23;

// Now we can convert the pair `(&mut a, &mut b)` to a mutable `TempInst` reference,
// and pass that to `run_twice`.
TempInst::call_with_mut((&mut a, &mut b), run_twice::<Bar>);

assert_eq!(a, 42 + 23 + 1 + 23);
assert_eq!(b, 23 + 1 + 1);
```

See the [source code](temp-inst/src/lib.rs) for more info.

## Similar crates

[dyn-context](https://crates.io/crates/dyn-context) is another solution for the same problem. The
main difference is that it focuses on user-defined types, whereas temp-inst defines a trait that is
already implemented for standard types but must currently be implemented manually for custom types.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
