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

pub(crate) fn read_codegen(ident: syn::Ident, generics: syn::Generics, span: Span, arrays: &[ReadInfo]) -> Result<TokenStream> {
    let mut body = quote!();

    for array in arrays {
        let ReadInfo {field_name, field_type, array_name, transpose} = array;

        let array_name_literal = syn::LitStr::new(array_name, span);


        // TODO: give some more error handling on this thing here
        // so the user knows what dataset was missing
        body = quote!(
            #body

            let #field_name = file.dataset(#array_name_literal)
                .map_err(|e| hdf5_derive::MissingDataset::from_field_name(#array_name_literal, e))?;
            let #field_name : #field_type = #field_name.read()
                .map_err(|e| hdf5_derive::SerializeArray::from_field_name(#array_name_literal, e))?;
        );

        // transpose the array if we need to 
        if *transpose {
            body = quote!(
                #body
                let #field_name = #field_name.reversed_axes();
            )
        }
    }

    // build the final return statement
    let punct : Punctuated<syn::Ident, syn::Token![,]> = arrays.iter().map(|arr| arr.field_name.clone()).collect();
    let return_statement = quote!(Ok(#ident { #punct }));

    let (imp, ty, wher) = generics.split_for_impl();

    // generate the full method implementation
    let full_impl = quote!(
        impl #imp #ident #ty #wher {
            fn read_hdf5(file: &hdf5_derive::File) -> Result<#ident, hdf5_derive::Error> {
                #body

                #return_statement
            }
        }
    );

    Ok(full_impl)
}
