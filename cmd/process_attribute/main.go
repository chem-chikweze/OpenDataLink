// Command process_attribute creates attribute embedding vectors and stores the
// attribute and the vectors in the Open Data Link database.
// chem
package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/DataIntelligenceCrew/OpenDataLink/internal/attributeembedding"
	"github.com/DataIntelligenceCrew/OpenDataLink/internal/config"
	"github.com/DataIntelligenceCrew/OpenDataLink/internal/vec32"
	"github.com/ekzhu/go-fasttext"
	_ "github.com/mattn/go-sqlite3"
)

const datasetsDir = "datasets"

type attributeNode struct {
	AttributeName string
	DatasetID     string
}

func attributeVector(ft *fasttext.FastText, m *attributeNode) ([]float32, error) {
	return attributeembedding.Vector(ft, m.AttributeName)
}

func main() {
	db, err := sql.Open("sqlite3", config.DatabasePath())
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	ft := fasttext.NewFastText(config.FasttextPath())
	defer ft.Close()

	tx, err := db.Begin()
	if err != nil {
		log.Fatal(err)
	}

	attributedataStmt, err := tx.Prepare(`
	INSERT INTO attributedata (
		attributename,
		dataset_id,
	)
	VALUES (?, ?)
	`)
	if err != nil {
		log.Fatal(err)
	}
	defer attributedataStmt.Close()

	vectorStmt, err := tx.Prepare(`
	INSERT INTO attribute_vectors (dataset_id, attribute_name, emb) VALUES (?, ?, ?)`)
	if err != nil {
		log.Fatal(err)
	}
	defer vectorStmt.Close()

	files, err := ioutil.ReadDir(datasetsDir)
	if err != nil {
		log.Fatal(err)
	}

	for _, f := range files {
		datasetID := f.Name()
		path := filepath.Join(datasetsDir, datasetID, "attribute.json")

		file, err := os.Open(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				log.Print(err)
				continue
			}
			log.Fatalf("dataset %v: %v", datasetID, err)
		}
		var m attributeNode
		if err := json.NewDecoder(file).Decode(&m); err != nil {
			log.Fatalf("dataset %v: %v", datasetID, err)
		}
		file.Close()

		_, err = attributedataStmt.Exec(
			m.AttributeName,
			m.DatasetID,
		)
		if err != nil {
			log.Fatalf("dataset %v: %v", datasetID, err)
		}

		emb, err := attributeVector(ft, &m)
		if err != nil && err != attributeembedding.ErrNoEmb {
			log.Fatalf("dataset %v: %v", datasetID, err)
		}
		_, err = vectorStmt.Exec(m.AttributeName, vec32.Bytes(emb))
		if err != nil {
			log.Fatalf("dataset %v: %v", datasetID, err)
		}
	}
	tx.Commit()
}
