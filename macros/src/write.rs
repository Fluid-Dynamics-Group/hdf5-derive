use proc_macro2::TokenStream;
use syn::Result;
use quote::quote;
use proc_macro2::Span;

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
