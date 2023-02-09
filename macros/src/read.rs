//use proc_macro::TokenStream;
use proc_macro2::TokenStream;
use syn::Result;
use quote::quote;
use proc_macro2::Span;
use syn::punctuated::Punctuated;

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
