mod read;
mod write;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput, Result};
use syn::spanned::Spanned;
use darling::{ast, FromDeriveInput, FromField};

#[proc_macro_derive(HDF5, attributes(hdf5))]
pub fn hdf5(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    derive(input)
        .map(Into::into)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[derive(Debug, FromDeriveInput)]
#[darling(supports(struct_any), attributes(hdf5))]
struct InputReceiver {
    /// The struct ident.
    #[allow(dead_code)]
    ident: syn::Ident,

    #[allow(dead_code)]
    generics: syn::Generics,

    /// Receives the body of the struct or enum. We don't care about
    /// struct fields because we previously told darling we only accept structs.
    data: ast::Data<(), FieldReceiver>,
}

#[derive(Debug, FromField)]
#[darling(attributes(hdf5))]
struct FieldReceiver {
    /// Get the ident of the field. For fields in tuple or newtype structs or
    /// enum bodies, this can be `None`.
    ident: Option<syn::Ident>,

    /// This magic field name pulls the type from the input.
    ty: syn::Type,

    #[darling(default)]
    /// whether or not to use `std::ops::Deref` on the field before 
    /// serializing the container
    transpose: bool
}

fn derive(input: DeriveInput) -> Result<TokenStream> {
    let receiver = InputReceiver::from_derive_input(&input).unwrap();

    // make sure we are dealing with a non-tuple / unit struct  struct
    let fields_information = match receiver.data {
        ast::Data::Enum(_) => unreachable!(),
        ast::Data::Struct(fields_with_style) => {
            match fields_with_style.style {
                ast::Style::Tuple | ast::Style::Unit => {
                    Err(
                        syn::parse::Error::new(input.span(), "Tuple / Unit structs are not accepted. Each field must be named")
                    )
                }
                ast::Style::Struct => {
                    Ok(
                        fields_with_style.fields
                    )
                }
            }
        }
    }?;

    // build the reading body:
    let read_data: Vec<read::ReadInfo> = fields_information
        .iter()
        .map(|rx: &FieldReceiver| {
            let field_name = rx.ident.clone().unwrap();
            let field_type = rx.ty.clone();
            let transpose = rx.transpose;
            let array_name = field_name.to_string();

            read::ReadInfo {field_name, field_type, transpose, array_name}

        }).collect();


    let read_impl = read::read_codegen(receiver.ident, receiver.generics, input.span(), &read_data)?;

    Ok(read_impl)
}
