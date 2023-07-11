var searchIndex = JSON.parse('{\
"hdf5_derive":{"doc":"hdf5-derive","t":"IYIYNNNNENNDDDGGGGGGNNNNINNNNNINNLLLLLLLLLLLLLLLLLLLLLLLLLALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLKLKLLLLLLLLLLLLLLLLLLLLLLKLKLDDDDDDDDDDDDDDDDDLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL","n":["ContainerRead","ContainerRead","ContainerWrite","ContainerWrite","CreateAttribute","CreateDataset","CreateGroup","DimensionMismatch","Error","FetchAttribute","FetchDataset","File","Group","LazyArray","LazyArray1","LazyArray2","LazyArray3","LazyArray4","LazyArray5","LazyArray6","MissingAttribute","MissingDataset","MissingDatatype","MissingGroup","ReadGroup","ReadSlice","SerializeArray","SerializeAttribute","WriteArray","WriteAttribute","WriteGroup","WriteSlice","WrongDatatype","access_plist","append","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","clone","clone","clone_into","clone_into","close","create","create_excl","create_group","create_plist","dataset","dataset","datasets","deref","deref","deref","error","fapl","fcpl","flush","fmt","fmt","fmt","fmt","free_space","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","group","groups","into","into","into","into","is_empty","is_read_only","iter_visit","iter_visit_default","len","link_exists","link_external","link_hard","link_soft","member_names","named_datatypes","new","new_dataset","new_dataset_builder","open","open_as","open_rw","provide","read_group","read_group","read_hdf5","relink","size","slice","source","to_owned","to_owned","to_string","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","unlink","userblock","with_options","write_group","write_group","write_hdf5","write_slice","CreateAttribute","CreateDataset","CreateGroup","DimensionMismatch","FetchAttribute","FetchDataset","MissingAttribute","MissingDataset","MissingDatatype","MissingGroup","ReadSlice","SerializeArray","SerializeAttribute","WriteArray","WriteAttribute","WriteSlice","WrongDatatype","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","borrow_mut","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","fmt","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","from_field_name","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","into","new","new","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","provide","source","source","source","source","source","source","source","source","source","source","source","source","source","source","source","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","to_string","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_from","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","try_into","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id","type_id"],"q":[[0,"hdf5_derive"],[141,"hdf5_derive::error"]],"d":["Provides methods for reading a struct’s contents from a …","","Provides methods for writing a struct’s contents to a …","","Could not create a attribute in a hdf5 file when writing","Could not create a dataset in a hdf5 file when writing","Could not create a group in a hdf5 file when writing","<code>crate::lazy_array::LazyArray</code> underlying <code>hdf5::Dataset</code> was …","General error type that provides helpful information on …","Attribute was missing from hdf5 file","Could fetch an existing dataset when writing file","HDF5 file object.","Represents the HDF5 group object.","An object that can be treated similar to an <code>ndarray::Array</code>,","one dimensional lazy array backed by HDF5 dataset","two dimensional lazy array backed by HDF5 dataset","three dimensional lazy array backed by HDF5 dataset","four dimensional lazy array backed by HDF5 dataset","five dimensional lazy array backed by HDF5 dataset","six dimensional lazy array backed by HDF5 dataset","Attribute was missing from hdf5 file","Error when attempting to read a dataset that does not …","Failed to fetch the datatype of a given dataset","Could not open an existing group in a hdf5 file when …","Defines how a given piece of data should be parsed. You …","Failed to read a slice of data from an HDF5 dataset","Error when attempting to read an array to a given struct …","Attribute was missing from hdf5 file","Failed to serialize an array to a hdf5 dataset","Could not create a attribute in a hdf5 file when writing","Defines how a given piece of data should be written. You …","Failed to write a slice of data to an HDF5 dataset","Datatype for <code>crate::LazyArray</code> was incorrect","Returns a copy of the file access property list.","Opens a file as read/write if exists, creates otherwise.","","","","","","","","","","","","","Closes the file and invalidates all open handles for …","Creates a file, truncates if exists.","Creates a file, fails if exists.","Create a new group in a file or group.","Returns a copy of the file creation property list.","provides access to the underlying dataset","Opens an existing dataset in the file or group.","Returns all datasets in the group, non-recursively","","","","Error types for handling HDF5 data with additional context …","A short alias for <code>access_plist()</code>.","A short alias for <code>create_plist()</code>.","Flushes the file to the storage medium.","","","","","Returns the free space in the file in bytes (or 0 if the …","Returns the argument unchanged.","","","","","","","","","","","","","Returns the argument unchanged.","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Opens an existing group in a file or group.","Returns all groups in the group, non-recursively","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Returns true if the container has no linked objects (or if …","Returns true if the file was opened in a read-only mode.","Visits all objects in the group","Visits all objects in the group using default …","Returns the number of objects in the container (or 0 if …","Check if a link with a given name exists in this file or …","Creates an external link.","Creates a hard link. Note: <code>target</code> and <code>link_name</code> are …","Creates a soft link.","Returns the names of all objects in the group, …","Returns all named types in the group, non-recursively","generate a dimension-checked and type-checked <code>LazyArray</code> …","Instantiates a new typed dataset builder.","Instantiates a new dataset builder.","Opens a file as read-only, file must exist.","Opens a file in a given mode.","Opens a file as read/write, file must exist.","","Given an hdf5 <code>hdf5::Group</code> and the name of the array we …","","read the contents of an HDF5 file to <code>Self</code>","Relinks an object. Note: <code>name</code> and <code>path</code> are relative to the …","Returns the file size in bytes (or 0 if the file handle is …","read a slice of data from the HDF5 dataset.","","","","","","","","","","","","","","","","","Removes a link to an object from this file or group.","Returns the userblock size in bytes (or 0 if the file …","Opens a file with custom file-access and file-creation …","Given an hdf5 <code>hdf5::Group</code> and the name of the array we …","","write the contents of a struct to an HDF5 file","write an array slice to a the underlying HDF5 dataset","Could not create a dataset in a hdf5 file when writing","Could not create a dataset in a hdf5 file when writing","Could not create a group in a hdf5 file when writing","<code>crate::lazy_array::LazyArray</code> underlying <code>hdf5::Dataset</code> was …","Could fetch an existing dataset when writing file","Could fetch an existing dataset when writing file","Could fetch an existing dataset when writing file","Error when attempting to read a dataset that does not …","Failed to fetch the datatype of a given dataset","Could not open an existing group in a hdf5 file when …","Failed to read a slice of data from an HDF5 dataset","Error when attempting to read an array to a given struct …","Failed to serialize the dataset to the correct type after …","Failed to serialize an array to a hdf5 dataset","Failed to serialize an attribute to a hdf5 dataset","Failed to write a slice of data to an HDF5 dataset","Datatype for <code>crate::LazyArray</code> was incorrect","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","Returns the argument unchanged.","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","construct an instance of this type using the name of the …","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","Calls <code>U::from(self)</code>.","constructor for this type","constructor for this type","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""],"i":[0,0,0,0,17,17,17,17,0,17,17,0,0,0,0,0,0,0,0,0,17,17,17,17,0,17,17,17,17,17,0,17,17,1,1,12,17,1,7,12,17,1,7,1,7,1,7,1,1,1,7,1,12,7,7,12,1,7,0,1,1,1,17,17,1,7,1,12,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,1,7,7,7,12,17,1,7,7,1,7,7,7,7,7,7,7,7,7,12,7,7,1,1,1,17,59,12,60,7,1,12,17,1,7,17,12,17,1,7,12,17,1,7,12,17,1,7,7,1,1,61,12,62,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,28,38,38,35,35,25,25,27,27,29,29,22,22,23,23,32,32,34,34,30,30,33,33,24,24,26,26,31,31,36,36,37,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37,28,38,35,25,27,29,22,23,32,34,30,33,24,26,31,36,37],"f":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[1,[[4,[2,3]]]],[[[6,[5]]],[[4,[1,3]]]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[1,1],[7,7],[[]],[[]],[1,[[4,[3]]]],[[[6,[5]]],[[4,[1,3]]]],[[[6,[5]]],[[4,[1,3]]]],[[7,8],[[4,[7,3]]]],[1,[[4,[9,3]]]],[[[12,[10,11]]],13],[[7,8],[[4,[13,3]]]],[7,[[4,[[15,[13,14]],3]]]],[[[12,[10,11]]]],[1,7],[7,16],0,[1,[[4,[2,3]]]],[1,[[4,[9,3]]]],[1,[[4,[3]]]],[[17,18],19],[[17,18],19],[[1,18],[[4,[20]]]],[[7,18],[[4,[20]]]],[1,21],[[]],[22,17],[23,17],[24,17],[25,17],[26,17],[27,17],[28,17],[29,17],[30,17],[31,17],[32,17],[33,17],[[]],[34,17],[35,17],[36,17],[37,17],[38,17],[[]],[[]],[[7,8],[[4,[7,3]]]],[7,[[4,[[15,[7,14]],3]]]],[[]],[[]],[[]],[[]],[7,39],[1,39],[[7,40,41,42],[[4,[3]]]],[[7,42],[[4,[3]]]],[7,21],[[7,8],39],[[7,8,8,8],[[4,[3]]]],[[7,8,8],[[4,[3]]]],[[7,8,8],[[4,[3]]]],[7,[[4,[[15,[43,14]],3]]]],[7,[[4,[[15,[44,14]],3]]]],[13,[[4,[[12,[10,11]],17]]]],[7,45],[7,46],[[[6,[5]]],[[4,[1,3]]]],[[[6,[5]],47],[[4,[1,3]]]],[[[6,[5]]],[[4,[1,3]]]],[48],[[7,8,39],[[4,[49,17]]]],[[7,8,39],[[4,[[12,[10,11]],17]]]],[7,[[4,[49,17]]]],[[7,8,8],[[4,[3]]]],[1,21],[[[12,[10,11]],[0,[[50,[11]],[52,[51]]]]],[[4,[[53,[10]],17]]]],[17,[[55,[54]]]],[[]],[[]],[[],43],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],56],[[],56],[[],56],[[],56],[[7,8],[[4,[3]]]],[1,21],[[],57],[[49,7,8,39,39],[[4,[17]]]],[[[12,[10,11]],7,8,39,39],[[4,[17]]]],[7,[[4,[17]]]],[[[12,[10,11]],[0,[[50,[11]],[52,[51]]]]],[[4,[17]]]],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[28,18],19],[[28,18],19],[[38,18],19],[[38,18],19],[[35,18],19],[[35,18],19],[[25,18],19],[[25,18],19],[[27,18],19],[[27,18],19],[[29,18],19],[[29,18],19],[[22,18],19],[[22,18],19],[[23,18],19],[[23,18],19],[[32,18],19],[[32,18],19],[[34,18],19],[[34,18],19],[[30,18],19],[[30,18],19],[[33,18],19],[[33,18],19],[[24,18],19],[[24,18],19],[[26,18],19],[[26,18],19],[[31,18],19],[[31,18],19],[[36,18],19],[[36,18],19],[[37,18],19],[[37,18],19],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[8,3],28],[[8,3],38],[[8,3],35],[[8,3],25],[[8,3],27],[[8,3],29],[[8,3],22],[[8,3],23],[[8,3],32],[[8,3],34],[[8,3],30],[[8,3],33],[[8,3],24],[[8,3],26],[[8,3],31],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[]],[[8,58,58],36],[[8,44],37],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[48],[28,[[55,[54]]]],[38,[[55,[54]]]],[35,[[55,[54]]]],[25,[[55,[54]]]],[27,[[55,[54]]]],[29,[[55,[54]]]],[22,[[55,[54]]]],[23,[[55,[54]]]],[32,[[55,[54]]]],[34,[[55,[54]]]],[30,[[55,[54]]]],[33,[[55,[54]]]],[24,[[55,[54]]]],[26,[[55,[54]]]],[31,[[55,[54]]]],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],43],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],4],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56],[[],56]],"c":[],"p":[[3,"File"],[3,"FileAccess"],[4,"Error"],[4,"Result"],[3,"Path"],[8,"AsRef"],[3,"Group"],[15,"str"],[3,"FileCreate"],[8,"H5Type"],[8,"Dimension"],[3,"LazyArray"],[3,"Dataset"],[3,"Global"],[3,"Vec"],[3,"Location"],[4,"Error"],[3,"Formatter"],[6,"Result"],[3,"Error"],[15,"u64"],[3,"WriteArray"],[3,"WriteAttribute"],[3,"MissingDatatype"],[3,"CreateGroup"],[3,"ReadSlice"],[3,"MissingGroup"],[3,"MissingDataset"],[3,"CreateAttribute"],[3,"MissingAttribute"],[3,"WriteSlice"],[3,"FetchDataset"],[3,"SerializeAttribute"],[3,"FetchAttribute"],[3,"CreateDataset"],[3,"DimensionMismatch"],[3,"WrongDatatype"],[3,"SerializeArray"],[15,"bool"],[4,"IterationOrder"],[4,"TraversalOrder"],[8,"Fn"],[3,"String"],[3,"Datatype"],[3,"DatasetBuilderEmpty"],[3,"DatasetBuilder"],[4,"OpenMode"],[3,"Demand"],[8,"Sized"],[8,"SliceArg"],[4,"Selection"],[8,"TryInto"],[6,"Array"],[8,"Error"],[4,"Option"],[3,"TypeId"],[3,"FileBuilder"],[15,"usize"],[8,"ReadGroup"],[8,"ContainerRead"],[8,"WriteGroup"],[8,"ContainerWrite"]]}\
}');
if (typeof window !== 'undefined' && window.initSearch) {window.initSearch(searchIndex)};
if (typeof exports !== 'undefined') {exports.searchIndex = searchIndex};