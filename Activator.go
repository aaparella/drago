package Drago

// Activator represents the activation function for a given layer
type Activator interface {
	// Apply calculates the activation of a given layer given
	// given previous layer activations
	Apply(int, int, float64) float64
	// Derivative is the calculation used to update weights during backprop
	Derivative(int, int, float64) float64
}
