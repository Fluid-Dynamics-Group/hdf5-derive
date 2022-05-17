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


#[derive(Debug, Clone, Copy, darling::FromMeta)]
#[darling(default)]
enum TransposeOpts {
    Read,
    Write,
    Both,
    None
}

impl TransposeOpts {
    fn transpose_read(&self) -> bool {
        match self {
            Self::Read | Self::Both => true,
            _ => false
        }
    }

    fn transpose_write(&self) -> bool {
        match self {
            Self::Write | Self::Both => true,
            _ => false
        }
    }
}

impl Default for TransposeOpts{
    fn default() -> Self {
        TransposeOpts::None
    }
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

    #[darling(default)]
    transpose: TransposeOpts
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
    transpose: Option<TransposeOpts>
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
            let transpose = rx.transpose.unwrap_or(receiver.transpose).transpose_read();

            let array_name = field_name.to_string();

            read::ReadInfo {field_name, field_type, transpose, array_name}

        }).collect();

    // build the writing body:
    let write_data: Vec<write::WriteInfo> = fields_information
        .iter()
        .map(|rx: &FieldReceiver| {
            let field_name = rx.ident.clone().unwrap();
            let field_type = rx.ty.clone();
            let transpose = rx.transpose.unwrap_or(receiver.transpose).transpose_write();
            let array_name = field_name.to_string();

            write::WriteInfo {field_name, field_type, transpose, array_name}

        }).collect();


    let read_impl = read::read_codegen(receiver.ident.clone(), receiver.generics.clone(), input.span(), &read_data)?;
    let write_impl = write::write_codegen(receiver.ident.clone(), receiver.generics.clone(), input.span(), &write_data)?;

    let impls = quote::quote!(
        #read_impl

        #write_impl
    );

    Ok(impls.into())
}
