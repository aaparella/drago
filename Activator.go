package Drago

type Activator interface {
	Apply(int, int, float64) float64
	Derivative(int, int, float64) float64
}
