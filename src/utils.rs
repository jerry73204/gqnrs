use crate::{
    common::*,
    data::{GqnExample, GqnFeature},
};

pub fn decode_image_on_example(
    example: GqnExample,
    feature_formats: HashMap<String, Option<ImageFormat>>,
) -> Fallible<GqnExample> {
    let new_example = example
        .into_iter()
        .map(|(name, feature)| {
            let format_opt = match feature_formats.get(&name) {
                Some(format) => format,
                None => return Ok((name, feature)),
            };

            let list = match feature {
                GqnFeature::BytesList(list) => list,
                _ => bail!("operation on non-bytes list is not supported"),
            };

            let new_list = list
                .into_iter()
                .map(|bytes| {
                    let array = decode_image(&bytes, *format_opt)?;
                    Ok(array)
                })
                .collect::<Fallible<Vec<_>>>()?;

            Ok((name, GqnFeature::DynamicImageList(new_list)))
        })
        .collect::<Fallible<GqnExample>>()?;

    Ok(new_example)
}

fn decode_image(bytes: &[u8], format_opt: Option<ImageFormat>) -> Fallible<DynamicImage> {
    let image_reader = format_opt
        .map(|format| {
            let reader = Cursor::new(bytes);
            Ok(ImageReader::with_format(reader, format))
        })
        .unwrap_or_else(|| {
            let reader = Cursor::new(bytes);
            ImageReader::new(reader).with_guessed_format()
        })?;
    let image = image_reader.decode()?;
    Ok(image)
}
