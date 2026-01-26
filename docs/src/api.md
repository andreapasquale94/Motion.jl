# API

```@index
```

This page lists the public API grouped by area. Each section is generated from the docstrings in the 
corresponding source files. If you are looking for a specific symbol, use the index above or your 
editor's search.

## Models 

### CR3BP 

```@autodocs
Modules = [Motion.CR3BP]
Public = true
Private = false
```

### ER3BP 

```@autodocs
Modules = [Motion.ER3BP]
Public = true
Private = false
```

### BCR4BP 

```@autodocs
Modules = [Motion.BCR4BP]
Public = true
Private = false
```

## Utilities

```@autodocs
Modules = [Motion]
Pages = [
    "models/utils.jl",
    "models/libration_points.jl",
]
Order = [:type, :function, :constant, :macro]
Public = true
Private = false
```

## Solutions

```@autodocs
Modules = [Motion]
Pages = [
    "solution.jl",
]
Order = [:type, :function, :constant, :macro]
Public = true
Private = false
```

## Measures

```@autodocs
Modules = [Motion]
Pages = [
    "models/measures.jl",
]
Order = [:type, :function, :constant, :macro]
Public = true
Private = false
```

## Optimization problem models

### Impulsive Multiple Shooting

```@autodocs
Modules = [Motion.ImpulsiveShooting]
Order = [:type, :function, :constant, :macro]
Public = true
```

### Constant Thrust Multiple Shooting

```@autodocs
Modules = [Motion.ConstThrustShooting]
Order = [:type, :function, :constant, :macro]
Public = true
```
