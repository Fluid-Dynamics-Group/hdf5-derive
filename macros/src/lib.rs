mod read;
mod write;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput, Result};
use syn::spanned::Spanned;
use darling::{ast, FromDeriveInput, FromField};

#[proc_macro_derive(ContainerRead, attributes(hdf5))]
pub fn container_read(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    read::derive_container_read(input)
        .map(Into::into)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro_derive(ContainerWrite, attributes(hdf5))]
pub fn container_write(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    write::derive_container_write(input)
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
        matches!(self, Self::Read | Self::Both)
    }

    fn transpose_write(&self) -> bool {
        matches!(self, Self::Write | Self::Both)
    }
}

impl Default for TransposeOpts{
    fn default() -> Self {
        TransposeOpts::None
    }
}

#[derive(Debug, Clone, darling::FromMeta, Default)]
#[darling(default)]
struct Rename {
    read: Option<String>,
    write: Option<String>,
    both: Option<String>,
}

impl Rename {
    /// name of the array if we are reading
    fn read_name_or_ident(&self, ident: &syn::Ident) -> String {
        self.both.as_ref().or(self.read.as_ref()).map(Into::into).unwrap_or_else(|| ident.to_string())
    }

    /// name of the array if we are writing
    fn write_name_or_ident(&self, ident: &syn::Ident) -> String {
        self.both.as_ref().or(self.write.as_ref()).map(Into::into).unwrap_or_else( || ident.to_string())
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
    transpose: TransposeOpts,

    #[darling(default)]
    mutate_on_write: bool,
}

#[derive(Debug, FromField, Clone)]
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
    transpose: Option<TransposeOpts>,

    #[darling(default)]
    /// whether or not to use `std::ops::Deref` on the field before 
    /// serializing the container
    rename: Rename,

    #[darling(default)]
    mutate_on_write: Option<bool>,
}

fn fields_from_input(input: &DeriveInput) -> Result<(InputReceiver, Vec<FieldReceiver>)> {
    let receiver = InputReceiver::from_derive_input(&input).unwrap();

    // make sure we are dealing with a non-tuple / unit struct  struct
    let fields = match receiver.data {
        ast::Data::Enum(_) => unreachable!(),
        ast::Data::Struct(ref fields_with_style) => {
            match fields_with_style.style {
                ast::Style::Tuple | ast::Style::Unit => {
                    Err(
                        syn::parse::Error::new(input.span(), "Tuple / Unit structs are not accepted. Each field must be named")
                    )
                }
                ast::Style::Struct => {
                    Ok(
                        fields_with_style.fields.clone()
                    )
                }
            }
        }
    }?;

    Ok((receiver, fields))
}
