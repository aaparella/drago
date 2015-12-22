package Drago

import "github.com/gonum/matrix/mat64"

type Network struct {
	Activators   []Activator
	Activations  []*mat64.Dense
	Weights      []*mat64.Dense
	Errors       []*mat64.Dense
	Layers       int
	LearningRate float64
	Iterations   int
	Err          Error
}

func New(learnRate float64, iterations int, topology []int, acts []Activator) *Network {
	net := &Network{
		LearningRate: learnRate,
		Iterations:   iterations,
		Activators:   make([]Activator, len(topology)),
		Activations:  make([]*mat64.Dense, len(topology)),
		Errors:       make([]*mat64.Dense, len(topology)),
		Weights:      make([]*mat64.Dense, len(topology)-1),
		Layers:       len(topology),
		Err:          MSE,
	}

	net.initActivations(topology)
	net.initWeights(topology)
	net.initActivators(acts)
	net.initErrors(topology)

	return net
}

func (n *Network) initActivations(topology []int) {
	for i, nodes := range topology {
		n.Activations[i] = mat64.NewDense(nodes, 1, nil)
	}
}

func (n *Network) initErrors(topology []int) {
	for i := 1; i < n.Layers; i++ {
		n.Errors[i] = mat64.NewDense(topology[i], 1, nil)
	}
}

func (n *Network) initWeights(topology []int) {
	n.Weights[0] = MatrixWithInitialValue(topology[1], topology[0], 1)
	for i := 1; i < n.Layers-1; i++ {
		n.Weights[i] = RandomMatrix(topology[i+1], topology[i])
	}
}

func (n *Network) initActivators(acts []Activator) {
	for i := 0; i < len(acts); i++ {
		n.Activators[i+1] = acts[i]
	}
}

func (n *Network) Learn(dataset [][][]float64) {
	for _, sample := range dataset {
		n.Forward(sample[0])
		n.Back(sample[1])
	}
}

func (n *Network) Forward(sample []float64) {
	n.Activations[0].SetCol(0, sample)
	for i := 0; i < len(n.Weights); i++ {
		n.Activations[i+1].Mul(n.Weights[i], n.Activations[i])
		if i != len(n.Weights)-1 {
			n.Activations[i+1].Apply(n.Activators[i+1].Apply, n.Activations[i+1])
		}
	}
}

func (n *Network) Back(label []float64) {
	// TODO
}
