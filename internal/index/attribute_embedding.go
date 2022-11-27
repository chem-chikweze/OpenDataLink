// chem
package index

import (
	"github.com/DataIntelligenceCrew/OpenDataLink/internal/database"
	"github.com/DataIntelligenceCrew/OpenDataLink/internal/vec32"
	"github.com/DataIntelligenceCrew/go-faiss"
)

// AttributeIndex is an index over the attribute embedding vectors.
type AttributeIndex struct {
	idx *faiss.IndexFlat
	// Maps ID of vector in index to dataset ID.
	idMap []string
}

// BuildAttributeEmbeddingIndex builds a MetadataIndex.
func BuildAttributeEmbeddingIndex(db *database.DB) (*MetadataIndex, error) {
	index, err := faiss.NewIndexFlatIP(300)
	if err != nil {
		return nil, err
	}

	rows, err := db.Query(`SELECT dataset_id, attribute_name, emb FROM attribute_vectors`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var idMap []string
	var vecs []float32

	for rows.Next() {
		var datasetID string
		var attributeName string
		var emb []byte

		if err := rows.Scan(&datasetID, &attributeName, &emb); err != nil {
			return nil, err
		}
		vec, err := vec32.FromBytes(emb)
		if err != nil {
			return nil, err
		}
		idMap = append(idMap, datasetID)
		vecs = append(vecs, vec...)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if err := index.Add(vecs); err != nil {
		return nil, err
	}

	return &MetadataIndex{index, idMap}, nil
}

// Delete frees the memory associated with the index.
func (idx *AttributeIndex) Delete() {
	idx.idx.Delete()
}

// Query queries the index with vec.
//
// Returns the dataset IDs of the (up to) k nearest neighbors and the
// corresponding cosine similarity, sorted by similarity.
func (idx *AttributeIndex) Query(vec []float32, k int64) ([]string, []float32, error) {
	dist, ids, err := idx.idx.Search(vec, k)
	if err != nil {
		return nil, nil, err
	}
	datasets := make([]string, 0, k)

	for _, id := range ids {
		if id == -1 {
			break
		}
		datasets = append(datasets, idx.idMap[id])
	}
	return datasets, dist[:len(datasets)], nil
}
