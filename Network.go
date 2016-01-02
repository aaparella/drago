// Feed Forward Artifical Neural Network Library
package Drago

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

type Network struct {
	Activators   []Activator
	Activations  []*mat64.Dense
	Weights      []*mat64.Dense
	Errors       []*mat64.Dense
	Topology     []int
	Layers       int
	LearningRate float64
	Iterations   int
	Err          Error
}

// Topology specifies number of hidden layers and nodes in each, as well as
// size of samples and labels (first and last values, respectively).
// Acts array should have one activator for each hidden layer
func New(learnRate float64, iterations int, topology []int, acts []Activator) *Network {
	net := &Network{
		LearningRate: learnRate,
		Iterations:   iterations,
		Activators:   make([]Activator, len(topology)),
		Activations:  make([]*mat64.Dense, len(topology)),
		Errors:       make([]*mat64.Dense, len(topology)),
		Weights:      make([]*mat64.Dense, len(topology)-1),
		Topology:     topology,
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
	for i := 0; i < n.Layers; i++ {
		n.Errors[i] = mat64.NewDense(topology[i], 1, nil)
	}
}

func (n *Network) initWeights(topology []int) {
	for i := 0; i < n.Layers-1; i++ {
		n.Weights[i] = randomMatrix(topology[i+1], topology[i])
	}
}

func (n *Network) initActivators(acts []Activator) {
	acts = append([]Activator{new(Linear)}, append(acts, new(Linear))...)
	for i := 0; i < len(acts); i++ {
		n.Activators[i] = acts[i]
	}
}

// Sample must have number of features specified by topology
func (n *Network) Predict(sample []float64) *mat64.Dense {
	n.Forward(sample)
	return n.Activations[n.Layers-1]
}

// Samples must have number of features and labels as specified by topology
// when constructing the network
func (n *Network) Learn(dataset [][][]float64) {
	fmt.Println("Learning...")
	for i := 0; i < n.Iterations; i++ {
		fmt.Println("=== Iteration ", i+1, " ===")
		for _, sample := range dataset {
			n.Forward(sample[0])
			n.Back(sample[1])
		}
	}
}

func (n *Network) Forward(sample []float64) {
	n.Activations[0].SetCol(0, sample)
	for i := 0; i < len(n.Weights); i++ {
		n.activateLayer(i)
	}
}

func (n *Network) activateLayer(layer int) {
	n.Activations[layer+1].Mul(n.Weights[layer], n.Activations[layer])
	n.Activations[layer+1].Apply(n.Activators[layer+1].Apply, n.Activations[layer+1])
}

func (n *Network) Back(label []float64) {
	n.calculateErrors(label)
	n.updateWeights()
}

func (n *Network) calculateErrors(label []float64) {
	actual := mat64.NewDense(len(label), 1, label)
	n.Errors[n.Layers-1].Sub(n.Activations[n.Layers-1], actual)
	for i := n.Layers - 2; i >= 0; i-- {
		n.calculateErrorForLayer(i)
	}
}

func (n *Network) calculateErrorForLayer(layer int) {
	n.Errors[layer].Mul(n.Weights[layer].T(), n.Errors[layer+1])
	n.Errors[layer].MulElem(n.Errors[layer], n.Activations[layer])
	mat := &mat64.Dense{}
	mat.Apply(n.Activators[layer].Derivative, n.Activations[layer])
	n.Errors[layer].MulElem(mat, n.Errors[layer])
}

func (n *Network) updateWeights() {
	for i := 0; i < n.Layers-1; i++ {
		mat := &mat64.Dense{}
		mat.Mul(n.Errors[i+1], n.Activations[i].T())
		mat.Scale(n.LearningRate, mat)
		n.Weights[i].Sub(n.Weights[i], mat)
	}
}
