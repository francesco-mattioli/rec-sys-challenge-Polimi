from Data_Handler.DataReader import DataReader

d=DataReader()
urm = d.load_augmented_binary_urm()
urm_df = d.csr_to_dataframe(urm)
print(len(urm_df["UserID"].unique()))
print("fatto")
urm,icm = d.pad_with_zeros_ICMandURM(urm)
print(len(urm_df["UserID"].unique()))