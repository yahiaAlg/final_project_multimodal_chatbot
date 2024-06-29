big datamining-operation: {
webscraping sources
curated datasets
specialized datasets

Github Repos: {
awesome prompts csv dataset
}
websites: {
prompthero
tensor-art
lexica-art
playground-ai
}
webscraping sources -> websites
curated datasets -> CC-Stories Dataset
curated datasets -> WikiSQL Dataset
specialized datasets -> Github Repos
specialized datasets -> Diffusiondb
specialized datasets -> Prompt Engineering and Responses Dataset

websites -> Data Processing (cleaning, normalisation, deduplication, annotation)
CC-Stories Dataset -> Data Processing (cleaning, normalisation, deduplication, annotation)
WikiSQL Dataset -> Data Processing (cleaning, normalisation, deduplication, annotation)
Github Repos -> Data Processing (cleaning, normalisation, deduplication, annotation)
Diffusiondb -> Data Processing (cleaning, normalisation, deduplication, annotation)
Prompt Engineering and Responses Dataset -> Data Processing (cleaning, normalisation, deduplication, annotation)

Classifications: {
text-to-image
text-to-audio

    text-to-text

    image-to-text
    audio-to-text

}

Data Processing (cleaning, normalisation, deduplication, annotation) -> Classifications

text-to-text-sub-Classifications: {
debugging prompts
question answer prompts
creative writing prompts
normal conversation or chatting prompts
coding prompts
}

Classifications.text-To-text -> text-to-text-sub-Classifications

image-to-text-sub-Classifications: {
description prompts
OCR text extraction prompts
Object identification prompts
Color Pallete extraction
}

Classifications.image-to-text -> image-to-text-sub-Classifications

text-to-audio-sub-Classifications: {
speech prompts
sound generation prompts
music generation prompts
}

Classifications.text-to-audio -> text-to-audio-sub-Classifications

audio-to-text-sub-Classifications: {
sound descrtiption
speech transcription
}

Classifications.audio-to-text -> audio-to-text-sub-Classifications

text-to-image-sub-Classifications: {
image generation prompts
image editing
image up-scaling prompts
image higher resolution-fix prompts
}
Classifications.text-to-image -> text-to-image-sub-Classifications
}
