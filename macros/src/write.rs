use proc_macro2::TokenStream;
use syn::Result;
use quote::quote;
use proc_macro2::Span;
use syn::punctuated::Punctuated;

pub(crate) struct WriteInfo {
    pub(crate) field_name: syn::Ident,
    pub(crate) field_type: syn::Type,
    pub(crate) array_name: String,
    pub(crate) transpose: bool,
}

pub(crate) fn write_codegen(ident: syn::Ident, generics: syn::Generics, span: Span, arrays: &[WriteInfo]) -> Result<TokenStream> {
    let mut body = quote!();

    for array in arrays {
        let WriteInfo {field_name, field_type, array_name, transpose} = array;

        let array_name_literal = syn::LitStr::new(&array_name, span);


        // TODO: give some more error handling on this thing here
        // so the user knows what dataset was missing
        body = quote!(
            #body

            let #field_name = file.new_dataset::<<#field_type as hdf5_derive::ArrayType>::Ty>()
                .shape(self.#field_name.shape())
                .create(#array_name_literal)
                .map_err(|e| hdf5_derive::CreateDataset::from_field_name(#array_name_literal, e))?;

            #field_name.write(&self.#field_name)
                .map_err(|e| hdf5_derive::WriteArray::from_field_name(#array_name_literal, e))?;
        );

        // transpose the array if we need to 
        if *transpose {
            body = quote!(
                #body
                let #field_name = #field_name.reversed_axes();
            )
        }
    }

    let (imp, ty, wher) = generics.split_for_impl();

    // generate the full method implementation
    let full_impl = quote!(
        impl #imp #ident #ty #wher {
            fn write_hdf5(&self, file: &hdf5_derive::File) -> Result<(), hdf5_derive::Error> {
                #body

                Ok(())
            }
        }
    );

    Ok(full_impl.into())
}
