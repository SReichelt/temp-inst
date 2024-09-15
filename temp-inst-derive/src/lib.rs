//! Derive macros for the `temp_inst` crate. These can be used to implement `temp_inst::TempRepr`
//! etc. for a struct or enum consisting entirely of items implementing `TempRepr`, identically to
//! the implementation for the corresponding tuple or `either::Either`.
//!
//! Using a custom type as a `TempRepr` instance is particularly useful when additional traits must
//! be implemented for the type.

use std::cell::Cell;

use proc_macro2::{Literal, TokenStream};
use punctuated::Punctuated;
use quote::{quote, ToTokens};
use syn::{spanned::Spanned, *};

/// Derives `temp_inst::TempRepr` for the given struct or enum.
///
/// `TempRepr::Shared` is defined as follows.
/// * `()` if the type is an empty struct.
/// * `T::Shared` if the type is a struct with exactly one field of type `T`.
/// * A tuple if the type is a struct with more than one field.
/// * `core::convert::Infallible` if the type is an empty enum.
/// * `()`, a single type, or a tuple if the type is an enum with a single variant, equivalently to
///   the corresponding struct.
/// * `either::Either<T0, T1>` if the type is an enum with two variants that would individually
///   result in `T0` and `T1`. This requires the "either" feature of the `temp_inst` crate to be
///   enabled.
/// * `Either<Either<T0, T1>, T2>` if the type is an enum with three variants, and so on.
///
/// In contrast to standard derive macros, `TempRepr` is implemented unconditionally if the type
/// has generic parameters. The reason for this behavior is that the parameters could be used as
/// arguments to `SelfRepr` or `TempRef` rather than directly.
#[proc_macro_derive(TempRepr)]
pub fn derive_temp_repr(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match TempReprImpl::new(&input) {
        Ok(TempReprImpl {
            shared,
            new_temp,
            get,
            ..
        }) => {
            let ident = &input.ident;
            let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

            let expanded = quote! {
                #[automatically_derived]
                #[allow(clippy::unused_unit)]
                unsafe impl #impl_generics temp_inst::TempRepr for #ident #ty_generics #where_clause {
                    type Shared<'__a> = #shared where Self: '__a;

                    #[allow(unused_variables)]
                    unsafe fn new_temp(obj: Self::Shared<'_>) -> Self {
                        #new_temp
                    }

                    fn get(&self) -> Self::Shared<'_> {
                        #get
                    }
                }
            };

            expanded.into()
        }

        Err(err) => err.into_compile_error().into(),
    }
}

/// Derives `temp_inst::TempReprMut` for the given struct or enum.
///
/// `TempReprMut::Mutable` is defined as follows.
/// * `()` if the type is an empty struct.
/// * `T::Mutable` if the type is a struct with exactly one field of type `T`.
/// * A tuple if the type is a struct with more than one field.
/// * `core::convert::Infallible` if the type is an empty enum.
/// * `()`, a single type, or a tuple if the type is an enum with a single variant, equivalently to
///   the corresponding struct.
/// * `either::Either<T0, T1>` if the type is an enum with two variants that would individually
///   result in `T0` and `T1`. This requires the "either" feature of the `temp_inst` crate to be
///   enabled.
/// * `Either<Either<T0, T1>, T2>` if the type is an enum with three variants, and so on.
///
/// In contrast to standard derive macros, `TempReprMut` is implemented unconditionally if the type
/// has generic parameters, except that `TempRepr` constraints are strengthened to `TempReprMut`.
/// The reason for this behavior is that the parameters could be used as arguments to `SelfRepr` or
/// `TempRef` rather than directly.
#[proc_macro_derive(TempReprMut)]
pub fn derive_temp_repr_mut(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match TempReprImpl::new(&input) {
        Ok(TempReprImpl {
            mutable,
            new_temp_mut,
            get_mut,
            get_mut_pinned,
            ..
        }) => {
            let ident = &input.ident;
            let mut generics = input.generics;
            strengthen_bounds(&mut generics, "TempRepr", "TempReprMut");
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

            let expanded = quote! {
                #[automatically_derived]
                #[allow(clippy::unused_unit)]
                unsafe impl #impl_generics temp_inst::TempReprMut for #ident #ty_generics #where_clause {
                    type Mutable<'__a> = #mutable where Self: '__a;

                    #[allow(unused_variables)]
                    unsafe fn new_temp_mut(obj: Self::Mutable<'_>) -> Self {
                        #new_temp_mut
                    }

                    fn get_mut(&mut self) -> Self::Mutable<'_> {
                        #get_mut
                    }

                    fn get_mut_pinned(self: core::pin::Pin<&mut Self>) -> Self::Mutable<'_> {
                        unsafe {
                            let temp = self.get_unchecked_mut();
                            #get_mut_pinned
                        }
                    }
                }
            };

            expanded.into()
        }

        Err(err) => err.into_compile_error().into(),
    }
}

/// Derives `temp_inst::TempReprMutChk` for the given struct or enum.
///
/// In contrast to standard derive macros, `TempReprMutChk` is implemented unconditionally if the
/// type has generic parameters, except that `TempRepr` or `TempReprMut` constraints are
/// strengthened to `TempReprMutChk`. The reason for this behavior is that the parameters could be
/// used as arguments to `SelfRepr` or `TempRef` rather than directly.
#[proc_macro_derive(TempReprMutChk)]
pub fn derive_temp_repr_mut_chk(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match TempReprImpl::new(&input) {
        Ok(TempReprImpl {
            swap_chk_data_type,
            swap_chk_data,
            ..
        }) => {
            let ident = &input.ident;
            let mut generics = input.generics;
            strengthen_bounds(&mut generics, "TempRepr", "TempReprMutChk");
            strengthen_bounds(&mut generics, "TempReprMut", "TempReprMutChk");
            let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

            let expanded = quote! {
                #[automatically_derived]
                #[allow(clippy::unused_unit)]
                unsafe impl #impl_generics temp_inst::TempReprMutChk for #ident #ty_generics #where_clause {
                    type SwapChkData = #swap_chk_data_type;

                    fn swap_chk_data(&self) -> Self::SwapChkData {
                        #swap_chk_data
                    }
                }
            };

            expanded.into()
        }

        Err(err) => err.into_compile_error().into(),
    }
}

struct TempReprImpl {
    shared: TokenStream,
    mutable: TokenStream,
    new_temp: TokenStream,
    new_temp_mut: TokenStream,
    get: TokenStream,
    get_mut: TokenStream,
    get_mut_pinned: TokenStream,
    swap_chk_data_type: TokenStream,
    swap_chk_data: TokenStream,
}

impl TempReprImpl {
    fn new(input: &DeriveInput) -> Result<Self> {
        match &input.data {
            Data::Struct(data) => Ok(Self::fields(
                |_, member| quote!(&self.#member),
                |_, member| quote!(&mut self.#member),
                |_, member| quote!(&mut temp.#member),
                |idx| {
                    if let Some(idx) = idx {
                        let idx = Literal::usize_unsuffixed(idx);
                        quote!(obj.#idx)
                    } else {
                        quote!(obj)
                    }
                },
                quote!(Self),
                &data.fields,
            )),
            Data::Enum(data) => {
                let remaining = Cell::new(data.variants.len());
                let make_arm = |arm_prefix: &TokenStream, mut body: TokenStream| {
                    for _ in 0..remaining.get() {
                        body = quote!(either::Either::Left(#body));
                    }
                    quote!(#arm_prefix #body,)
                };
                let mut variant_iter = data.variants.iter();
                if let Some(first_variant) = variant_iter.next() {
                    remaining.set(remaining.get() - 1);
                    let (
                        TempReprImpl {
                            mut shared,
                            mut mutable,
                            mut new_temp,
                            mut new_temp_mut,
                            mut get,
                            mut get_mut,
                            mut get_mut_pinned,
                            mut swap_chk_data_type,
                            mut swap_chk_data,
                        },
                        arm_prefix,
                    ) = Self::variant(first_variant);
                    get = make_arm(&arm_prefix, get);
                    get_mut = make_arm(&arm_prefix, get_mut);
                    get_mut_pinned = make_arm(&arm_prefix, get_mut_pinned);
                    swap_chk_data = make_arm(&arm_prefix, swap_chk_data);
                    for variant in variant_iter {
                        remaining.set(remaining.get() - 1);
                        let (
                            TempReprImpl {
                                shared: variant_shared,
                                mutable: variant_mutable,
                                new_temp: variant_new_temp,
                                new_temp_mut: variant_new_temp_mut,
                                get: variant_get,
                                get_mut: variant_get_mut,
                                get_mut_pinned: variant_get_mut_pinned,
                                swap_chk_data_type: variant_swap_chk_data_type,
                                swap_chk_data: variant_swap_chk_data,
                            },
                            variant_arm_prefix,
                        ) = Self::variant(variant);
                        shared = quote!(either::Either<#shared, #variant_shared>);
                        mutable = quote!(either::Either<#mutable, #variant_mutable>);
                        swap_chk_data_type = quote!(either::Either<#swap_chk_data_type, #variant_swap_chk_data_type>);
                        new_temp = quote!(match obj {
                            either::Either::Left(obj) => #new_temp,
                            either::Either::Right(obj) => #variant_new_temp,
                        });
                        new_temp_mut = quote!(match obj {
                            either::Either::Left(obj) => #new_temp_mut,
                            either::Either::Right(obj) => #variant_new_temp_mut,
                        });
                        get.extend(make_arm(
                            &variant_arm_prefix,
                            quote!(either::Either::Right(#variant_get)),
                        ));
                        get_mut.extend(make_arm(
                            &variant_arm_prefix,
                            quote!(either::Either::Right(#variant_get_mut)),
                        ));
                        get_mut_pinned.extend(make_arm(
                            &variant_arm_prefix,
                            quote!(either::Either::Right(#variant_get_mut_pinned)),
                        ));
                        swap_chk_data.extend(make_arm(
                            &variant_arm_prefix,
                            quote!(either::Either::Right(#variant_swap_chk_data)),
                        ));
                    }
                    Ok(TempReprImpl {
                        shared,
                        mutable,
                        new_temp,
                        new_temp_mut,
                        get: quote!(match self {#get}),
                        get_mut: quote!(match self {#get_mut}),
                        get_mut_pinned: quote!(match temp {#get_mut_pinned}),
                        swap_chk_data_type,
                        swap_chk_data: quote!(match self {#swap_chk_data}),
                    })
                } else {
                    Ok(TempReprImpl {
                        shared: quote!(core::convert::Infallible),
                        mutable: quote!(core::convert::Infallible),
                        new_temp: quote!(match obj {}),
                        new_temp_mut: quote!(match obj {}),
                        get: quote!(match *self {}),
                        get_mut: quote!(match *self {}),
                        get_mut_pinned: quote!(match *temp {}),
                        swap_chk_data_type: quote!(core::convert::Infallible),
                        swap_chk_data: quote!(match *self {}),
                    })
                }
            }
            Data::Union(data) => Err(Error::new(
                data.union_token.span(),
                "TempRepr cannot be derived for unions",
            )),
        }
    }

    fn fields(
        self_member: impl Fn(usize, &Member) -> TokenStream,
        self_member_mut: impl Fn(usize, &Member) -> TokenStream,
        pinned_member_mut: impl Fn(usize, &Member) -> TokenStream,
        obj_member: impl Fn(Option<usize>) -> TokenStream,
        prefix: TokenStream,
        fields: &Fields,
    ) -> Self {
        if fields.len() == 1 {
            let field = fields.iter().next().unwrap();
            let member = fields.members().next().unwrap();
            let self_member = self_member(0, &member);
            let self_member_mut = self_member_mut(0, &member);
            let pinned_member_mut = pinned_member_mut(0, &member);
            let single = Self::single(
                self_member,
                self_member_mut,
                pinned_member_mut,
                obj_member(None),
                &field.ty,
            );
            let new_temp = single.new_temp;
            let new_temp_mut = single.new_temp_mut;
            TempReprImpl {
                new_temp: quote!(#prefix {#member: #new_temp}),
                new_temp_mut: quote!(#prefix {#member: #new_temp_mut}),
                ..single
            }
        } else {
            let multi = fields
                .iter()
                .zip(fields.members())
                .enumerate()
                .map(|(idx, (field, member))| {
                    let self_member = self_member(idx, &member);
                    let self_member_mut = self_member_mut(idx, &member);
                    let pinned_member_mut = pinned_member_mut(idx, &member);
                    let obj_member = obj_member(Some(idx));
                    let single = Self::single(
                        self_member,
                        self_member_mut,
                        pinned_member_mut,
                        obj_member,
                        &field.ty,
                    );
                    let new_temp = single.new_temp;
                    let new_temp_mut = single.new_temp_mut;
                    TempReprImpl {
                        new_temp: quote!(#member: #new_temp),
                        new_temp_mut: quote!(#member: #new_temp_mut),
                        ..single
                    }
                })
                .collect::<Vec<_>>();
            let shared = multi.iter().map(|single| &single.shared);
            let mutable = multi.iter().map(|single| &single.mutable);
            let new_temp = multi.iter().map(|single| &single.new_temp);
            let new_temp_mut = multi.iter().map(|single| &single.new_temp_mut);
            let get = multi.iter().map(|single| &single.get);
            let get_mut = multi.iter().map(|single| &single.get_mut);
            let get_mut_pinned = multi.iter().map(|single| &single.get_mut_pinned);
            let swap_chk_data_type = multi.iter().map(|single| &single.swap_chk_data_type);
            let swap_chk_data = multi.iter().map(|single| &single.swap_chk_data);
            TempReprImpl {
                shared: quote!((#(#shared,)*)),
                mutable: quote!((#(#mutable,)*)),
                new_temp: quote!(#prefix {#(#new_temp,)*}),
                new_temp_mut: quote!(#prefix {#(#new_temp_mut,)*}),
                get: quote!((#(#get,)*)),
                get_mut: quote!((#(#get_mut,)*)),
                get_mut_pinned: quote!((#(#get_mut_pinned,)*)),
                swap_chk_data_type: quote!((#(#swap_chk_data_type,)*)),
                swap_chk_data: quote!((#(#swap_chk_data,)*)),
            }
        }
    }

    fn single(
        member: TokenStream,
        member_mut: TokenStream,
        pinned_member_mut: TokenStream,
        obj: TokenStream,
        ty: &Type,
    ) -> Self {
        TempReprImpl {
            shared: quote!(<#ty as temp_inst::TempRepr>::Shared<'__a>),
            mutable: quote!(<#ty as temp_inst::TempReprMut>::Mutable<'__a>),
            new_temp: quote!(<#ty as temp_inst::TempRepr>::new_temp(#obj)),
            new_temp_mut: quote!(<#ty as temp_inst::TempReprMut>::new_temp_mut(#obj)),
            get: quote!(<#ty as temp_inst::TempRepr>::get(#member)),
            get_mut: quote!(<#ty as temp_inst::TempReprMut>::get_mut(#member_mut)),
            get_mut_pinned: quote!(<#ty as temp_inst::TempReprMut>::get_mut_pinned(core::pin::Pin::new_unchecked(#pinned_member_mut))),
            swap_chk_data_type: quote!(<#ty as temp_inst::TempReprMutChk>::SwapChkData),
            swap_chk_data: quote!(<#ty as temp_inst::TempReprMutChk>::swap_chk_data(#member)),
        }
    }

    fn variant(variant: &Variant) -> (Self, TokenStream) {
        let variant_ident = &variant.ident;
        let members = variant.fields.members();
        let fields = match &variant.fields {
            Fields::Named(_) => {
                quote!({#(#members,)*})
            }
            Fields::Unnamed(_) => {
                let members = members
                    .enumerate()
                    .map(|(idx, member)| Ident::new(&format!("temp{idx}"), member.span()));
                quote!((#(#members,)*))
            }
            Fields::Unit => TokenStream::new(),
        };
        let member = |idx: usize, member: &Member| {
            if matches!(variant.fields, Fields::Named(_)) {
                member.to_token_stream()
            } else {
                Ident::new(&format!("temp{idx}"), member.span()).into_token_stream()
            }
        };
        let result = Self::fields(
            member,
            member,
            member,
            |idx| {
                if let Some(idx) = idx {
                    let idx = Literal::usize_unsuffixed(idx);
                    quote!(obj.#idx)
                } else {
                    quote!(obj)
                }
            },
            quote!(Self::#variant_ident),
            &variant.fields,
        );
        (result, quote!(Self::#variant_ident #fields =>))
    }
}

fn strengthen_bounds(generics: &mut Generics, src: &str, dst: &str) {
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            strengthen_type_param_bounds(&mut type_param.bounds, src, dst);
        }
    }

    if let Some(where_clause) = &mut generics.where_clause {
        for predicate in &mut where_clause.predicates {
            if let WherePredicate::Type(type_predicate) = predicate {
                strengthen_type_param_bounds(&mut type_predicate.bounds, src, dst);
            }
        }
    }
}

fn strengthen_type_param_bounds(
    bounds: &mut Punctuated<TypeParamBound, Token![+]>,
    src: &str,
    dst: &str,
) {
    for bound in bounds {
        if let TypeParamBound::Trait(trait_bound) = bound {
            if let Some(segment) = trait_bound.path.segments.last_mut() {
                if segment.ident == src {
                    segment.ident = Ident::new(dst, segment.ident.span());
                }
            }
        }
    }
}
