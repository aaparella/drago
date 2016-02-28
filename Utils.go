package Drago

import (
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
)

func randomMatrix(r, c int) *mat64.Dense {
	rand.Seed(time.Now().UnixNano())
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.Float64() / 2.0
	}
	return mat64.NewDense(r, c, data)
}
