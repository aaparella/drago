package Drago

import "github.com/gonum/matrix/mat64"

type Network struct {
	Activators   []Activator
	Activations  []*mat64.Dense
	Weights      []*mat64.Dense
	LearningRate float64
	Iterations   int
}

func New(learnRate float64, iterations int, topology []int, acts []Activator) *Network {
	net := &Network{
		LearningRate: learnRate,
		Iterations:   iterations,
		Activators:   make([]Activator, len(topology)-2),
		Activations:  make([]*mat64.Dense, len(topology)),
		Weights:      make([]*mat64.Dense, len(topology)-2),
	}

	for i, nodes := range topology {
		net.Activations[i] = mat64.NewDense(nodes, 1, nil)
	}

	for i := 1; i < len(topology)-1; i++ {
		net.Weights[i-1] = mat64.NewDense(topology[i], topology[i+1], nil)
		net.Activators[i-1] = acts[i-1]
	}

	return net
}

func Learn(dataset [][]float64, labels []float64) {

}

func Forward(sample []float64) {
}

func Back(label []float64) {

}
