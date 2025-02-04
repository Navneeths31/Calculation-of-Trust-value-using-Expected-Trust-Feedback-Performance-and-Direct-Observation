# Trust-calculation

The trust value of an entity can be calculated based on some metrices such as,

1. Performance (P) 
2. Expected Trust (E) 
3. Direct Observation (O)
4. Feedback (F)

With the help of the general formula, we can see that 

Tc = P + E + O + F
and that
Trust Value (TV) = log (log (Tc/(1-Tc)))  

We created Python functions to perform the above project and created a sample data sheet to access the data and write to it. Usage of tanh (Hyperbolic tangent) was used to calculate the output, the mean value of Trust.
