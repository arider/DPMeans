using Distributions: MvNormal
using LinearAlgebra: norm, I
using Statistics: mean
using Random

"""
Implementation of hard version of dpmm. DPMeans for small variance asymptotics
approach taken in this paper: 'Revisiting k-means: New Algorithms via Bayesian
Nonparametrics'.
"""

mutable struct Mixture
    components::Array{Distribution}
    weights::Array{Float64}
end

function likelihood(mixture::Mixture, point)
    sum([pdf(mixture.components[i], point) * mixture.weights[i] for i in 1:length(mixture.components)])
end

function sample(mixture::Mixture, n_samples=1)
    # Sample how many times each component is randomly selected.
    sampled_components = rand(Multinomial(n_samples, mixture.weights))
    samples = []
    for (ci, count) in enumerate(sampled_components)
        # sample from the component the number of times it was selected.
        for col in eachcol(rand(mixture.components[ci], count))
            push!(samples, col)
        end
    end
    return samples
end

mutable struct DPMeans
    lambda::Float64
    kernels::Array{MvNormal, 1}
    weights::Array{Float64, 1}
end

function DPMeans(lambda=0.5)
    lambda = lambda
    kernels = Array{MvNormal, 1}()
    DPMeans(lambda, kernels, Array{Float64, 1}())
end

"""
Train a DPMeans on the provided data. TODO: make this efficient (sub O(n)).
"""
function fit!(model::DPMeans, data, n_iterations=5, lambda_c=20, min_cluster_size=1)
    # If this is the first time fit was called, set the kernel to a random
    update_lambda=false   # point in the data.
    if size(model.kernels, 1) == 0
        push!(model.kernels, MvNormal(data[rand(1:size(data, 1)), :], model.lambda))
        push!(model.weights, 1.0)
    end

    labels = [0 for i in 1:size(data, 1)]
    for iteration in 1:n_iterations
        # For all observations.
        for i in 1:size(data, 1)
            # Calculate distances to all cluster centers.
            #   There has got to be a nicer way to do this.
            row_distances = [data[i, :] .- model.kernels[ki].μ for ki in 1:size(model.kernels, 1)]
            row_distances = [norm(row_distances[i, :]) for i in 1:size(row_distances, 1)]

            # Find the min and check if it is > lambda.
            min_index = argmin(row_distances)
            min_value = row_distances[min_index]
            if min_value > model.lambda
                # Add a new cluster to model.kernels.
                push!(model.kernels, MvNormal(vec(data[i, :]), model.lambda))
                push!(model.weights, 0.0)
            end
            labels[i] = min_index
        end

        # Move the cluster means to reflect the labels.
        # Remove any clusters that don't have points assigned to them.
        keep_inds = []
        for label in 1:maximum(labels)
            cluster_data = data[labels .== label, :]
            covariance = model.lambda
            if size(cluster_data, 1) == 1 || sum(covariance) == 0.0
                covariance = I * model.lambda
            end
            model.kernels[label] = MvNormal(vec(mean(cluster_data, dims=1)), covariance)
            model.weights[label] = size(cluster_data, 1)

            # We can't have clusters with one thing in them.
            if length(cluster_data) > min_cluster_size
                push!(keep_inds, label)
            end

        end
        model.kernels = model.kernels[keep_inds]
        model.weights = model.weights[keep_inds] / sum(model.weights[keep_inds])

        # Now make a mixture
        mixture = Mixture(model.kernels, model.weights)
    end
end


function get_labels(model::DPMeans, data)
    labels = [0 for i in 1:size(data, 1)]
    for i in 1:size(data, 1)
       # Calculate distances to all cluster centers.
       #   There has got to be a nicer way to do this.
        row_distances = [data[i, :] .- model.kernels[ki].μ for ki in 1:size(model.kernels, 1)]
        row_distances = [norm(row_distances[i, :]) for i in 1:size(row_distances, 1)]

        # Find the min and check if it is > lambda.
        min_index = argmin(row_distances)
        min_value = row_distances[min_index]
        labels[i] = min_index
    end
    labels
end

function get_mixture(model::DPMeans)
    return Mixture(model.kernels, model.weights)
end

