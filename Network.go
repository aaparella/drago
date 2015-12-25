package Drago

import "github.com/gonum/matrix/mat64"

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
		n.Activations[i] = mat64.NewDense(nodes+1, 1, nil)
	}
}

func (n *Network) initErrors(topology []int) {
	for i := 1; i < n.Layers; i++ {
		n.Errors[i] = mat64.NewDense(topology[i]+1, 1, nil)
	}
}

func (n *Network) initWeights(topology []int) {
	n.Weights[0] = matrixWithInitialValue(topology[1]+1, topology[0]+1, 1)
	for i := 1; i < n.Layers-1; i++ {
		n.Weights[i] = randomMatrix(topology[i+1]+1, topology[i]+1)
	}
}

func (n *Network) initActivators(acts []Activator) {
	for i := 0; i < len(acts); i++ {
		n.Activators[i+1] = acts[i]
	}
}

func (n *Network) Predict(sample []float64) *mat64.Dense {
	n.Forward(sample)
	return n.Activations[n.Layers-1]
}

func (n *Network) Learn(dataset [][][]float64) {
	for i := 0; i < n.Iterations; i++ {
		for _, sample := range dataset {
			n.Forward(sample[0])
			n.Back(sample[1])
		}
	}
}

func (n *Network) Forward(sample []float64) {
	sample = append([]float64{1}, sample...)
	n.Activations[0].SetCol(0, sample)
	for i := 0; i < len(n.Weights); i++ {
		n.activateLayer(i)
	}
}

func (n *Network) activateLayer(layer int) {
	n.Activations[layer].Set(0, 0, 1)
	n.Activations[layer+1].Mul(n.Weights[layer], n.Activations[layer])
	if layer != len(n.Weights)-1 {
		n.Activations[layer+1].Apply(n.Activators[layer+1].Apply, n.Activations[layer+1])
	}
}

func (n *Network) Back(label []float64) {
	n.calculateErrors(label)
	n.updateWeights()
}

func (n *Network) calculateErrors(label []float64) {
	label = append([]float64{1}, label...)
	actual := mat64.NewDense(len(label), 1, label)
	n.Errors[n.Layers-1].Sub(n.Activations[n.Layers-1], actual)
	for i := n.Layers - 2; i > 0; i-- {
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
