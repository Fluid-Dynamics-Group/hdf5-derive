//use proc_macro::TokenStream;
use proc_macro2::TokenStream;
use syn::{DeriveInput, Result};
use syn::spanned::Spanned;
use quote::quote;
use proc_macro2::Span;
use syn::punctuated::Punctuated;
use super::{fields_from_input, FieldReceiver};

pub(crate) struct ReadInfo {
    pub(crate) field_name: syn::Ident,
    pub(crate) field_type: syn::Type,
    pub(crate) array_name: String,
    pub(crate) transpose: bool,
}

pub(crate) fn read_codegen(ident: syn::Ident, span: Span, arrays: &[ReadInfo]) -> Result<TokenStream> {
    let mut body = quote!();


    for array_or_attribute in arrays {
        let ReadInfo { field_name, field_type, array_name, transpose } = array_or_attribute;

        let name = syn::LitStr::new(&array_name, span);

        body = quote!(
            #body
            let #field_name : #field_type = hdf5_derive::ReadGroup::read_group(group, #name, #transpose)?;
        );
    }

    // build the final return statement
    let punct : Punctuated<syn::Ident, syn::Token![,]> = arrays.iter().map(|arr| arr.field_name.clone()).collect();
    let return_statement = quote!(Ok(#ident { #punct }));

    // generate the full method implementation
    let full_impl = quote!(
        #body

        #return_statement
    );

    Ok(full_impl)
}

pub(crate) fn derive_container_read(input: DeriveInput) -> Result<TokenStream> {
    let (receiver, fields_information) = fields_from_input(&input)?;

    // build the reading body:
    let read_data: Vec<ReadInfo> = fields_information
        .iter()
        .map(|rx: &FieldReceiver| {
            let field_name = rx.ident.clone().unwrap();
            let field_type = rx.ty.clone();
            let transpose = rx.transpose.unwrap_or(receiver.transpose).transpose_read();

            let array_name = rx.rename.read_name_or_ident(&field_name);

            ReadInfo {field_name, field_type, transpose, array_name }

        }).collect();


    let read_impl = read_codegen(receiver.ident.clone(), input.span(), &read_data)?;

    let (imp, ty, wher) = receiver.generics.split_for_impl();
    let ident = receiver.ident.clone();

    let output = quote::quote!(
        impl #imp hdf5_derive::ContainerRead for #ident #ty #wher {
            fn read_hdf5(group: &hdf5_derive::Group) -> Result<Self, hdf5_derive::Error> {
                #read_impl
            }
        }
    ).into();

    Ok(output)
}
