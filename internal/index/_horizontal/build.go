package horizontal

import (
	"errors"
	"fmt"
	"os"

	"opendatalink/internal/database"

	"github.com/justinfargnoli/go-fasttext"
	"github.com/justinfargnoli/lshforest/pkg"
)

const dimensionCount = fasttext.Dim
const hashTableCount = 256
const hashValuePerHashTableCount = 64

func fastTextPath() (string, error) {
	path := os.Getenv("FAST_TEXT_DB")
	if path == "" {
		return "", errors.New("'FAST_TEXT_DB' enviroment variable is not set")
	}
	return path, nil
}

// BuildMetadataIndex builds a LSH index using github.com/fnargesian/simhash-lsh
func BuildMetadataIndex(db *database.DB) (Index, error) {
	indexBuilder := NewIndexBuilder(dimensionCount)

	metadataIterator, err := db.NewMetadataIterator()
	if err != nil {
		return Index{}, err
	}
	for metadataIterator.HasNext() {
		metadata, err := metadataIterator.Row()
		if err != nil {
			return Index{}, err
		}
		indexBuilder.InsertMetadata(metadata)
	}
	if err := metadataIterator.End(); err != nil {
		return Index{}, nil
	}

	return indexBuilder.ToIndex(), nil
}

// IndexBuilder is a write only wrapper of simhashlsh.CosineLsh
type IndexBuilder struct {
	index    *lshforest.LSHForest
	fastText fasttext.FastText
}

// NewIndexBuilder constructs an IndexBuilder
//
// dimension, hashTableCount, hashValuePerHashTableCount  of
// NewIndexBuilder(dimension, hashTableCount, hashValuePerHashTableCount)
// map to simhash.NewCosinLsh(dim, l m)'s dim, l, and m respectivly
func NewIndexBuilder(dimension uint) IndexBuilder {
	path, err := fastTextPath()
	if err != nil {
		panic(err)
	}
	return IndexBuilder{
		index:    lshforest.New(5, dimension, dimension, lshforest.Cosine),
		fastText: *fasttext.New(path),
	}
}

// ToIndex coverts the IndexBuilder to an Index
func (indexBuilder IndexBuilder) ToIndex() Index {
	return Index{indexBuilder.index, indexBuilder.fastText}
}

// Insert adds the embeddingVector and id to the index
func (indexBuilder IndexBuilder) Insert(point *Point) {
	indexBuilder.index.Insert(&point.EmbeddingVector, point.ID())
}

// InsertZip zips the embeddingVectors and IDs array into a one dimensional
// array of (embeddingVector []float64, ID string) tuples which are then added
// to the index
func (indexBuilder IndexBuilder) InsertZip(embeddingVectors *[][]float64, datasetID string, IDs *[]string) {
	if len(*embeddingVectors) != len(*IDs) {
		panic(fmt.Sprintf("(len(embeddingVectors) = %v) != (len(IDs) = %v)", len(*embeddingVectors), len(*IDs)))
	}
	for i := range *embeddingVectors {
		indexBuilder.Insert(&Point{(*embeddingVectors)[i], datasetID, (*IDs)[i]})
	}
}

// Point is a point in the simhashlsh.CosineLsh Index
type Point struct {
	EmbeddingVector []float64
	DatasetID       string
	Value           string
}

// ID returns the ID to be passed to the third parameter of Index.Insert()
func (point *Point) ID() string {
	return point.DatasetID + point.Value
}

// InsertMetadataRows adds metadataRows to a simhashlsh.CosineLsh index
func (indexBuilder IndexBuilder) InsertMetadataRows(metadataRows *[]database.Metadata) error {
	for _, v := range *metadataRows {
		indexBuilder.InsertMetadata(&v)
	}
	return nil
}

// InsertMetadata adds one row of the metadata to the index
func (indexBuilder IndexBuilder) InsertMetadata(metadata *database.Metadata) {
	indexBuilder.InsertName(metadata)
	indexBuilder.InsertDescription(metadata)
	indexBuilder.InsertCategories(metadata)
	indexBuilder.InsertTags(metadata)
}

// InsertName adds Metadata.Name to index
func (indexBuilder IndexBuilder) InsertName(metadata *database.Metadata) {
	nameEmbeddingVector, err := NameEmbeddingVector(metadata, &indexBuilder.fastText)
	if err != nil {
		return
	}
	indexBuilder.Insert(&Point{nameEmbeddingVector, metadata.DatasetID, metadata.Name})
}

// InsertDescription adds Metadata.Description to index
func (indexBuilder IndexBuilder) InsertDescription(metadata *database.Metadata) {
	descriptionEmbeddingVectors, err :=
		DescriptionEmbeddingVectors(metadata, &indexBuilder.fastText)
	if err != nil {
		return
	}
	descriptionSplit := metadata.DescriptionSplit()
	indexBuilder.InsertZip(&descriptionEmbeddingVectors, metadata.DatasetID, &descriptionSplit)
}

// InsertCategories adds Metadata.Categories to index
func (indexBuilder IndexBuilder) InsertCategories(metadata *database.Metadata) {
	categoriesEmbeddingVectors, err :=
		CategoriesEmbeddingVectors(metadata, &indexBuilder.fastText)
	if err != nil {
		return
	}
	indexBuilder.InsertZip(&categoriesEmbeddingVectors, metadata.DatasetID, &metadata.Categories)
}

// InsertTags adds Metadata.Tags to index
func (indexBuilder IndexBuilder) InsertTags(metadata *database.Metadata) {
	tagsEmbeddingVectors, err := TagsEmbeddingVectors(metadata, &indexBuilder.fastText)
	if err != nil {
		return
	}
	indexBuilder.InsertZip(&tagsEmbeddingVectors, metadata.DatasetID, &metadata.Tags)
}

// NameEmbeddingVector returns the embedding vector which represents
// Metadata.Name
// []float64 == nil when an embedding vector does not exist for Metadata.Name
func NameEmbeddingVector(metadata *database.Metadata, fastText *fasttext.FastText) ([]float64, error) {
	nameSplit := metadata.NameSplit()
	embeddingVector, err := fastText.MultiWordEmbeddingVector(nameSplit)
	if err != nil {
		return nil, err
	}
	return embeddingVector, nil
}

// DescriptionEmbeddingVectors returns an array of embedding vectors which
// represent the words of Metadata.Description
func DescriptionEmbeddingVectors(metadata *database.Metadata, fastText *fasttext.FastText) ([][]float64, error) {
	descriptionSplit := metadata.DescriptionSplit()
	var descriptionEmbeddingVector [][]float64
	for _, v := range descriptionSplit {
		wordEmbeddingVector, err := fastText.EmbeddingVector(v)
		if err != nil {
			return nil, err
		}
		descriptionEmbeddingVector =
			append(descriptionEmbeddingVector, wordEmbeddingVector)
	}

	return descriptionEmbeddingVector, nil
}

// AttributionEmbeddingVectors returns an array of embedding vectors which
// represent the words of Metadata.Attribution
func AttributionEmbeddingVectors(metadata *database.Metadata, fastText *fasttext.FastText) ([][]float64, error) {
	attributionSplit := metadata.AttributionSplit()
	var attributionEmbeddingVector [][]float64
	for _, v := range attributionSplit {
		wordEmbeddingVector, err := fastText.EmbeddingVector(v)
		if err != nil {
			return nil, err
		}
		attributionEmbeddingVector =
			append(attributionEmbeddingVector, wordEmbeddingVector)
	}

	return attributionEmbeddingVector, nil
}

// CategoriesEmbeddingVectors returns an array of embedding vectors which
// represent the words of Metadata.Categories
func CategoriesEmbeddingVectors(metadata *database.Metadata, fastText *fasttext.FastText) ([][]float64, error) {
	var categoriesEmbeddingVector [][]float64
	for _, v := range metadata.Categories {
		wordEmbeddingVector, err := fastText.EmbeddingVector(v)
		if err != nil {
			return nil, err
		}
		categoriesEmbeddingVector =
			append(categoriesEmbeddingVector, wordEmbeddingVector)
	}

	return categoriesEmbeddingVector, nil
}

// TagsEmbeddingVectors returns an array of embedding vectors which
// represent the words of Metadata.Tags
func TagsEmbeddingVectors(metadata *database.Metadata, fastText *fasttext.FastText) ([][]float64, error) {
	var tagsEmbeddingVector [][]float64
	for _, v := range metadata.Tags {
		wordEmbeddingVector, err := fastText.EmbeddingVector(v)
		if err != nil {
			return nil, err
		}
		tagsEmbeddingVector = append(tagsEmbeddingVector, wordEmbeddingVector)
	}

	return tagsEmbeddingVector, nil
}