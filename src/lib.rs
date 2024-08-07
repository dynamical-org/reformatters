pub mod binary_round;
pub mod gfs;
pub mod http;
pub mod output;
use std::{collections::HashMap, error::Error, mem::size_of_val};

use anyhow::Result;
use backon::{ExponentialBuilder, Retryable};
use chrono::{DateTime, TimeDelta, Utc};
use futures::future::join_all;
use output::{PutPayload, PutResult, Storage};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DataVariable {
    pub name: &'static str,
    pub long_name: &'static str,
    pub units: &'static str,
    pub grib_variable_name: &'static str,
    pub dtype: &'static str,
    pub statistics_approximate: Value,
}

#[derive(Debug, Clone)]
pub struct DataDimension {
    pub name: &'static str,
    pub long_name: &'static str,
    pub units: &'static str,
    pub dtype: &'static str,
    pub extra_metadata: HashMap<&'static str, &'static str>,
    pub statistics_approximate: Value,
}

#[derive(Debug, Clone)]
pub struct AnalysisDataset {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub url: &'static str,
    pub spatial_coverage: &'static str,
    pub spatial_resolution: &'static str,
    pub attribution: &'static str,

    pub time_start: DateTime<Utc>,
    pub time_end: DateTime<Utc>,
    pub time_step: TimeDelta,
    pub time_chunk_size: usize,

    pub longitude_start: f64,
    pub longitude_end: f64,
    pub longitude_step: f64,
    pub longitude_chunk_size: usize,

    pub latitude_start: f64,
    pub latitude_end: f64,
    pub latitude_step: f64,
    pub latitude_chunk_size: usize,

    pub data_dimensions: Vec<DataDimension>,
    pub data_variables: Vec<DataVariable>,
}

#[derive(Debug, Clone)]
pub struct AnalysisRunConfig {
    pub dataset: AnalysisDataset,
    pub data_variable: DataVariable,
    pub time_coordinates: Arc<Vec<DateTime<Utc>>>, // Arc because this struct is cloned often and this vec can be big
    pub latitude_coordinates: Vec<f64>,
    pub longitude_coordinates: Vec<f64>,
    pub time_start: DateTime<Utc>,
    pub time_end: DateTime<Utc>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn num_chunks(size: usize, chunk_size: usize) -> usize {
    ((size as f64) / (chunk_size as f64)).ceil() as usize
}

async fn write_bytes(
    file_path: &str,
    bytes: Vec<u8>,
    object_store: Storage,
    dest_root_path: &str,
) -> Result<()> {
    let payload: PutPayload = bytes.into();
    (|| async {
        do_upload(
            object_store.clone(),
            format!("{dest_root_path}/{file_path}"),
            payload.clone(),
        )
        .await
    })
    .retry(&ExponentialBuilder::default())
    .await?;

    Ok(())
}

async fn write_json(
    file_path: &str,
    json_value: &Value,
    object_store: Storage,
    dest_root_path: &str,
) -> Result<()> {
    let json_string = serde_json::to_string_pretty(json_value)?;
    write_bytes(
        file_path,
        json_string.as_bytes().to_vec(),
        object_store,
        dest_root_path,
    )
    .await
}

fn to_value<T: Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).unwrap()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrCompressorMetadata {
    blocksize: usize,
    clevel: usize,
    cname: &'static str,
    id: &'static str,
    shuffle: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrFilterMetadata {}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrArrayMetadata {
    chunks: Vec<usize>,
    compressor: Option<ZarrCompressorMetadata>,
    dtype: &'static str,
    fill_value: Option<&'static str>,
    filters: Option<Vec<ZarrFilterMetadata>>,
    order: &'static str,
    shape: Vec<usize>,
    zarr_format: usize,
}

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrVariableAttributeMetadata {
    _ARRAY_DIMENSIONS: Vec<&'static str>,
    long_name: &'static str,
    units: &'static str,
    grid_mapping: &'static str,
    statistics_approximate: Value,
}

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrDimensionAttributeMetadata {
    _ARRAY_DIMENSIONS: Vec<&'static str>,
    long_name: &'static str,
    units: &'static str,
    statistics_approximate: Value,
}

async fn write_array_metadata(
    field_name: &str,
    mut zmetadata: HashMap<String, serde_json::Value>,
    zarr_array_json: &serde_json::Value,
    zarr_attribute_json: &serde_json::Value,
    store: Storage,
    dest_root_path: &str,
) -> Result<HashMap<String, serde_json::Value>> {
    zmetadata.insert(format!("{field_name}/.zarray"), to_value(&zarr_array_json));
    zmetadata.insert(
        format!("{field_name}/.zattrs"),
        to_value(&zarr_attribute_json),
    );

    write_json(
        &format!("{field_name}/.zarray"),
        zarr_array_json,
        store.clone(),
        dest_root_path,
    )
    .await?;
    write_json(
        &format!("{field_name}/.zattrs"),
        zarr_attribute_json,
        store.clone(),
        dest_root_path,
    )
    .await?;

    Ok(zmetadata.clone())
}

impl AnalysisRunConfig {
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub async fn write_zarr_metadata(&self, store: Storage, dest_root_path: &str) -> Result<()> {
        let data_variable_compressor_metadata = ZarrCompressorMetadata {
            blocksize: 0,
            clevel: 5,
            cname: "zstd",
            id: "blosc",
            shuffle: 1,
        };
        let order = "C";
        let fill_value = "NaN";
        let zarr_format = 2;

        let zgroup = json!({"zarr_format": 2});

        let mut zmetadata: HashMap<String, serde_json::Value> = HashMap::new();

        let time_shape_size = self.time_coordinates.len();
        let latitude_shape_size = self.latitude_coordinates.len();
        let longitude_shape_size = self.longitude_coordinates.len();

        // Data variable metadata
        for data_variable in self.dataset.data_variables.clone() {
            let zarr_array_metadata = ZarrArrayMetadata {
                chunks: [
                    self.dataset.time_chunk_size,
                    self.dataset.latitude_chunk_size,
                    self.dataset.longitude_chunk_size,
                ]
                .to_vec(),
                compressor: Some(data_variable_compressor_metadata.clone()),
                dtype: data_variable.dtype,
                fill_value: Some(fill_value),
                filters: None,
                order,
                shape: [time_shape_size, latitude_shape_size, longitude_shape_size].to_vec(),
                zarr_format,
            };

            let zarr_attribute_metadata = ZarrVariableAttributeMetadata {
                _ARRAY_DIMENSIONS: ["time", "latitude", "longitude"].to_vec(),
                long_name: data_variable.long_name,
                units: data_variable.units,
                grid_mapping: "spatial_ref",
                statistics_approximate: data_variable.statistics_approximate,
            };

            zmetadata = write_array_metadata(
                data_variable.name,
                zmetadata.clone(),
                &to_value(&zarr_array_metadata),
                &to_value(&zarr_attribute_metadata),
                store.clone(),
                dest_root_path,
            )
            .await?;
        }

        let data_dimension_compressor_metadata = ZarrCompressorMetadata {
            blocksize: 0,
            clevel: 5,
            cname: "zstd",
            id: "blosc",
            shuffle: 1,
        };

        for data_dimension in self.dataset.data_dimensions.clone() {
            let shape_size = match data_dimension.name {
                "time" => time_shape_size,
                "latitude" => latitude_shape_size,
                "longitude" => longitude_shape_size,
                &_ => todo!(),
            };

            let fill_value = match data_dimension.name {
                "time" => None,
                &_ => Some("NaN"),
            };

            let zarr_array_metadata = ZarrArrayMetadata {
                chunks: [shape_size].to_vec(),
                compressor: Some(data_dimension_compressor_metadata.clone()),
                dtype: data_dimension.dtype,
                fill_value,
                filters: None,
                order,
                shape: [shape_size].to_vec(),
                zarr_format,
            };

            let zarr_attribute_metadata = ZarrDimensionAttributeMetadata {
                _ARRAY_DIMENSIONS: [data_dimension.name].to_vec(),
                long_name: data_dimension.long_name,
                units: data_dimension.units,
                statistics_approximate: data_dimension.statistics_approximate,
            };

            let mut zarr_attribute_metadata_value = to_value(&zarr_attribute_metadata);
            let zarr_attribute_metadata_with_extra_fields =
                zarr_attribute_metadata_value.as_object_mut().unwrap();
            for (key, value) in data_dimension.extra_metadata {
                zarr_attribute_metadata_with_extra_fields.insert(key.to_string(), json!(value));
            }

            zmetadata = write_array_metadata(
                data_dimension.name,
                zmetadata.clone(),
                &to_value(&zarr_array_metadata),
                &to_value(&zarr_attribute_metadata_with_extra_fields),
                store.clone(),
                dest_root_path,
            )
            .await?;
        }

        // For rioxarray support
        let spatial_ref_array_metadata = ZarrArrayMetadata {
            chunks: [].to_vec(),
            compressor: None,
            dtype: "<i8",
            fill_value: None,
            filters: None,
            order: "C",
            shape: [].to_vec(),
            zarr_format: 2,
        };
        let spatial_ref_attribute_metadata = json!({
            "_ARRAY_DIMENSIONS": [],
            "crs_wkt": "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AXIS[\"Latitude\",NORTH],AXIS[\"Longitude\",EAST],AUTHORITY[\"EPSG\",\"4326\"]]",
            "geographic_crs_name": "WGS 84",
            "grid_mapping_name": "latitude_longitude",
            "horizontal_datum_name": "World Geodetic System 1984",
            "inverse_flattening": 298.257_223_563,
            "longitude_of_prime_meridian": 0.0,
            "prime_meridian_name": "Greenwich",
            "reference_ellipsoid_name": "WGS 84",
            "semi_major_axis": 6_378_137.0,
            "semi_minor_axis": 6_356_752.314_245_179,
            "spatial_ref": "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AXIS[\"Latitude\",NORTH],AXIS[\"Longitude\",EAST],AUTHORITY[\"EPSG\",\"4326\"]]"

        });
        zmetadata = write_array_metadata(
            "spatial_ref",
            zmetadata.clone(),
            &to_value(&spatial_ref_array_metadata),
            &spatial_ref_attribute_metadata,
            store.clone(),
            dest_root_path,
        )
        .await?;

        let dataset_zattrs = json!({
            "id": self.dataset.id,
            "name": self.dataset.name,
            "description": self.dataset.description,
            "attribution": self.dataset.attribution,
            "time_domain": format!("{} to {}", self.dataset.time_start, self.dataset.time_end),
            "time_resolution": format!("{} hour", self.dataset.time_step.num_hours()),
            "spatial_domain": self.dataset.spatial_coverage,
            "spatial_resolution": self.dataset.spatial_resolution,
            "coordinates": "time latitude longitude spatial_ref",
        });

        zmetadata.insert(".zattrs".to_string(), dataset_zattrs.clone());
        zmetadata.insert(".zgroup".to_string(), zgroup.clone());

        write_json(
            ".zmetadata",
            &json!({"metadata": to_value(&zmetadata), "zarr_consolidated_format": 1}),
            store.clone(),
            dest_root_path,
        )
        .await?;
        write_json(".zattrs", &dataset_zattrs, store.clone(), dest_root_path).await?;
        write_json(".zgroup", &zgroup, store.clone(), dest_root_path).await?;

        Ok(())
    }

    pub async fn write_dimension_coordinates(
        &self,
        store: Storage,
        dest_root_path: &str,
    ) -> Result<()> {
        let time_values: Vec<i64> = self
            .time_coordinates
            .iter()
            .map(DateTime::timestamp)
            .collect();

        let time_coords_future = write_bytes(
            "time/0",
            blosc_compress(&time_values),
            store.clone(),
            dest_root_path,
        );

        let latitude_coords_future = write_bytes(
            "latitude/0",
            blosc_compress(&self.latitude_coordinates),
            store.clone(),
            dest_root_path,
        );

        let longitude_coords_future = write_bytes(
            "longitude/0",
            blosc_compress(&self.longitude_coordinates),
            store.clone(),
            dest_root_path,
        );

        // rioxarray wants a coordinate with a single value called spatial_ref.
        // All of the coordinate reference system information is stored as attributes on this array
        let spatial_ref_coords_future =
            write_bytes("spatial_ref/0", [0].to_vec(), store.clone(), dest_root_path);

        join_all([
            time_coords_future,
            latitude_coords_future,
            longitude_coords_future,
            spatial_ref_coords_future,
        ])
        .await;

        Ok(())
    }
}

pub async fn do_upload(store: Storage, chunk_path: String, bytes: PutPayload) -> Result<PutResult> {
    let mut put_result_result = store.put(&chunk_path.into(), bytes).await;

    // A little nonsense to ignore deeply nested error if the ETag isn't
    // present on the response. The put still succeeds in this case.
    if let Err(e) = &put_result_result {
        if let Some(source) = e.source() {
            if let Some(source) = source.source() {
                if source.to_string().starts_with("ETag Header missing") {
                    put_result_result = Ok(PutResult {
                        e_tag: None,
                        version: None,
                    });
                }
            }
        }
    }

    Ok(put_result_result?)
}

pub fn blosc_compress<T>(data: &[T]) -> Vec<u8> {
    let element_size = size_of_val(&data[0]);

    let context = blosc::Context::new()
        .compressor(blosc::Compressor::Zstd)
        .unwrap()
        .typesize(Some(element_size))
        .clevel(blosc::Clevel::L5)
        .shuffle(blosc::ShuffleMode::Byte);

    context.compress(data).into()
}
