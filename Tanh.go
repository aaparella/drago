package Drago

import "math"

type Tanh struct {
}

func (t *Tanh) Apply(r, c int, val float64) float64 {
	return (1 - math.Exp(-2*val)) / (1 + math.Exp(-2*val))
}

func (t *Tanh) Derivative(r, c int, val float64) float64 {
	return 1 - (math.Pow((math.Exp(2*val)-1)/(math.Exp(2*val)+1), 2))
}
