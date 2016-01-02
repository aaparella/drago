package Drago

import "github.com/gonum/matrix/mat64"

type Criterion interface {
	Apply(*mat64.Dense, *mat64.Dense) float64
	Derivative(*mat64.Dense, *mat64.Dense) *mat64.Dense
}
