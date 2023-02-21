use proc_macro2::TokenStream;
use syn::{DeriveInput, Result};
use syn::spanned::Spanned;
use quote::quote;
use proc_macro2::Span;
use super::{fields_from_input, FieldReceiver};

pub(crate) struct WriteInfo {
    pub(crate) field_name: syn::Ident,
    pub(crate) array_name: String,
    pub(crate) transpose: bool,
    pub(crate) mutate_on_write: bool,
}

pub(crate) fn write_codegen(span: Span, arrays: &[WriteInfo]) -> Result<TokenStream> {
    let mut body = quote!();

    for array_or_attribute in arrays {
        let WriteInfo { field_name, array_name, transpose, mutate_on_write } = array_or_attribute;

        let name = syn::LitStr::new(&array_name, span);

        body = quote!(
            #body
            
            hdf5_derive::WriteGroup::write_group(&self.#field_name, &file, #name, #transpose, #mutate_on_write)?;
        );
    }

    // generate the full method implementation
    let full_impl = quote!(
        #body

        Ok(())
    );

    Ok(full_impl)
}

pub(crate) fn derive_container_write(input: DeriveInput) -> Result<TokenStream> {
    let (receiver, fields_information) = fields_from_input(&input)?;

    // build the writing body:
    let write_data: Vec<WriteInfo> = fields_information
        .iter()
        .map(|rx: &FieldReceiver| {
            let field_name = rx.ident.clone().unwrap();
            let transpose = rx.transpose.unwrap_or(receiver.transpose).transpose_write();
            let array_name = rx.rename.write_name_or_ident(&field_name);

            let mutate_on_write = rx.mutate_on_write.unwrap_or(receiver.mutate_on_write);

            WriteInfo {field_name, transpose, array_name, mutate_on_write}

        }).collect();

    //Ok(combine_impls(receiver.ident, receiver.generics, read_impl, write_impl).into());

    let write_impl = write_codegen(input.span(), &write_data)?;

    let (imp, ty, wher) = receiver.generics.split_for_impl();
    let ident = receiver.ident.clone();

    let output = quote::quote!(
        impl #imp hdf5_derive::ContainerWrite for #ident #ty #wher {
            fn write_hdf5(&self, file: &hdf5_derive::Group) -> Result<(), hdf5_derive::Error> {
                #write_impl
            }
        }
    ).into();

    Ok(output)
}
