import time
import numpy as np
import random
from genetic_kernel import *
from run_DQN import METADATA


def ga_kernel(N_satellite, N_target, sat_samplings, sat_attitudes, sat_in_darks,
              sat_vectors, sat_positions, sat_available, tar_prioritys, tar_vectors, tar_positions):
    popSize = METADATA['popSize']
    eliteSize = METADATA['eliteSize']
    chrom_size = N_satellite
    crossoverRate = METADATA['crossoverRate']
    mutationRate = METADATA['mutationRate']
    num_generations = METADATA['generations']

    #  input array
    cuda_chromosomes = cuda.to_device(np.zeros(shape=(popSize, chrom_size), dtype=np.int32))
    cuda_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))

    cuda_sorted_chromosomes = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_tmp_fitnesses = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_fitnessTotal = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_rouletteWheel = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_popSize = cuda.to_device(np.array([popSize], dtype=np.int32))
    cuda_eliteSize = cuda.to_device(np.array([eliteSize], dtype=np.int32))
    cuda_crossoverRate = cuda.to_device(np.array([crossoverRate], dtype=np.float64))
    cuda_mutationRate = cuda.to_device(np.array([mutationRate], dtype=np.float64))
    cuda_N_target = cuda.to_device(np.array([N_target], dtype=np.int32))
    cuda_sat_samplings = cuda.to_device(sat_samplings)
    cuda_sat_attitudes = cuda.to_device(sat_attitudes)
    cuda_sat_in_darks = cuda.to_device(sat_in_darks)
    cuda_sat_vectors = cuda.to_device(sat_vectors)
    cuda_sat_positions = cuda.to_device(sat_positions)
    cuda_sat_available = cuda.to_device(sat_available)
    cuda_tar_prioritys = cuda.to_device(tar_prioritys)
    cuda_tar_vectors = cuda.to_device(tar_vectors)
    cuda_tar_positions = cuda.to_device(tar_positions)
    cuda_is_adaptive = cuda.to_device(np.array([METADATA['adaptive']]))

    start = time.perf_counter()

    threads_per_block = 256
    blocks_per_grid = (popSize + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid, threads_per_block = 4, 256
    print(blocks_per_grid, threads_per_block)

    # states
    np.random.seed(METADATA["seed"])
    state_seeds = np.random.rand(3)
    print(state_seeds)

    states = []
    for i in range(len(state_seeds)):
        states.append(
            create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=METADATA["seed"] + np.random.rand()))

    init_population[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_N_target, states[0])

    for i in range(num_generations + 1):

        eval_genomes_kernel[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses, cuda_popSize,
                                                                cuda_N_target, cuda_sat_samplings, cuda_sat_attitudes,
                                                                cuda_sat_in_darks, cuda_sat_vectors, cuda_sat_positions,
                                                                cuda_sat_available, cuda_tar_prioritys,
                                                                cuda_tar_vectors, cuda_tar_positions)

        if i < num_generations:
            sort_chromosomes[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses,
                                                                 cuda_sorted_chromosomes,
                                                                 cuda_sorted_fitnesses, cuda_fitnessTotal)

            # Crossover And Mutation
            crossover[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_sorted_chromosomes,
                                                          cuda_sorted_fitnesses, cuda_rouletteWheel,
                                                          states[1], cuda_popSize, cuda_eliteSize, cuda_fitnessTotal,
                                                          cuda_crossoverRate, cuda_is_adaptive)

            mutation[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_eliteSize, cuda_tmp_fitnesses,
                                                         cuda_N_target, states[2], cuda_mutationRate, cuda_is_adaptive)

    end = time.perf_counter()
    show_fitness_pairs = []
    chromosomes = cuda_chromosomes.copy_to_host()
    fitnesses = cuda_fitnesses.copy_to_host()

    for i in range(len(chromosomes)):
        show_fitness_pairs.append([chromosomes[i], fitnesses[i]])
    fitnesses = list(reversed(sorted(fitnesses)))  # fitnesses now in descending order
    show_sorted_pairs = list(reversed(sorted(show_fitness_pairs, key=lambda x: x[1])))
    best_allocation = show_sorted_pairs[0][0]
    best_fitness = show_sorted_pairs[0][1]
    return best_allocation, best_fitness[0], end - start


def iea_kernel(N_satellite, N_target, sat_samplings, sat_attitudes, sat_in_darks,
               sat_vectors, sat_positions, sat_available, tar_prioritys, tar_vectors, tar_positions):
    popSize = METADATA['popSize']
    eliteSize = METADATA['eliteSize']
    chrom_size = N_satellite
    crossoverRate = [8, 0.75, 0.7, 0.6]
    mutationRate = [0.1, 0.05, 0.025, 0.01]
    num_generations = METADATA['generations']

    #  input array
    cuda_chromosomes1 = cuda.to_device(np.zeros(shape=(popSize, chrom_size), dtype=np.int32))
    cuda_chromosomes2 = cuda.to_device(np.zeros(shape=(popSize, chrom_size), dtype=np.int32))
    cuda_chromosomes3 = cuda.to_device(np.zeros(shape=(popSize, chrom_size), dtype=np.int32))
    cuda_chromosomes4 = cuda.to_device(np.zeros(shape=(popSize, chrom_size), dtype=np.int32))
    cuda_fitnesses1 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_fitnesses2 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_fitnesses3 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_fitnesses4 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))

    cuda_sorted_chromosomes1 = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_chromosomes2 = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_chromosomes3 = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_chromosomes4 = cuda.to_device(np.zeros([popSize, chrom_size], dtype=np.int32))
    cuda_sorted_fitnesses1 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_sorted_fitnesses2 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_sorted_fitnesses3 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_sorted_fitnesses4 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))
    cuda_tmp_fitnesses1 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_tmp_fitnesses2 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_tmp_fitnesses3 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_tmp_fitnesses4 = cuda.to_device(np.zeros([popSize, 1], dtype=np.float64))  # only for mutation
    cuda_fitnessTotal1 = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_fitnessTotal2 = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_fitnessTotal3 = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_fitnessTotal4 = cuda.to_device(np.zeros(shape=1, dtype=np.float64))
    cuda_rouletteWheel1 = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_rouletteWheel2 = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_rouletteWheel3 = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_rouletteWheel4 = cuda.to_device(np.zeros(shape=popSize, dtype=np.float64))
    cuda_crossoverRate1 = cuda.to_device(np.array([crossoverRate[0]], dtype=np.float64))
    cuda_crossoverRate2 = cuda.to_device(np.array([crossoverRate[1]], dtype=np.float64))
    cuda_crossoverRate3 = cuda.to_device(np.array([crossoverRate[2]], dtype=np.float64))
    cuda_crossoverRate4 = cuda.to_device(np.array([crossoverRate[3]], dtype=np.float64))
    cuda_mutationRate1 = cuda.to_device(np.array([mutationRate[0]], dtype=np.float64))
    cuda_mutationRate2 = cuda.to_device(np.array([mutationRate[1]], dtype=np.float64))
    cuda_mutationRate3 = cuda.to_device(np.array([mutationRate[2]], dtype=np.float64))
    cuda_mutationRate4 = cuda.to_device(np.array([mutationRate[3]], dtype=np.float64))
    cuda_popSize = cuda.to_device(np.array([popSize], dtype=np.int32))
    cuda_eliteSize = cuda.to_device(np.array([eliteSize], dtype=np.int32))
    cuda_N_target = cuda.to_device(np.array([N_target], dtype=np.int32))
    cuda_sat_samplings = cuda.to_device(sat_samplings)
    cuda_sat_attitudes = cuda.to_device(sat_attitudes)
    cuda_sat_in_darks = cuda.to_device(sat_in_darks)
    cuda_sat_vectors = cuda.to_device(sat_vectors)
    cuda_sat_positions = cuda.to_device(sat_positions)
    cuda_sat_available = cuda.to_device(sat_available)
    cuda_tar_prioritys = cuda.to_device(tar_prioritys)
    cuda_tar_vectors = cuda.to_device(tar_vectors)
    cuda_tar_positions = cuda.to_device(tar_positions)
    cuda_is_adaptive = cuda.to_device(np.array([METADATA['adaptive']]))

    start = time.perf_counter()

    threads_per_block = 256
    blocks_per_grid = (popSize + (threads_per_block - 1)) // threads_per_block
    blocks_per_grid, threads_per_block = 4, 256

    # states
    np.random.seed(METADATA['seed'])
    state_seeds = np.random.rand(6)
    states = []
    for i in range(len(state_seeds)):
        states.append(
            create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=METADATA["seed"] + np.random.rand()))

    for cuda_chromosomes in [cuda_chromosomes1, cuda_chromosomes2, cuda_chromosomes3, cuda_chromosomes4]:
        init_population[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_N_target, states[0])

    for i in range(num_generations + 1):
        for cuda_chromosomes, cuda_fitnesses in zip(
                [cuda_chromosomes1, cuda_chromosomes2, cuda_chromosomes3, cuda_chromosomes4],
                [cuda_fitnesses1, cuda_fitnesses2, cuda_fitnesses3, cuda_fitnesses4]):
            eval_genomes_kernel[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses, cuda_popSize,
                                                                    cuda_N_target, cuda_sat_samplings,
                                                                    cuda_sat_attitudes,
                                                                    cuda_sat_in_darks, cuda_sat_vectors,
                                                                    cuda_sat_positions,
                                                                    cuda_sat_available, cuda_tar_prioritys,
                                                                    cuda_tar_vectors, cuda_tar_positions)

        if i < num_generations:
            for cuda_chromosomes, cuda_fitnesses, cuda_sorted_chromosomes, cuda_sorted_fitnesses, cuda_fitnessTotal, cuda_rouletteWheel, cuda_crossoverRate, cuda_tmp_fitnesses, cuda_mutationRate in \
                    zip(
                        [cuda_chromosomes1, cuda_chromosomes2, cuda_chromosomes3, cuda_chromosomes4],
                        [cuda_fitnesses1, cuda_fitnesses2, cuda_fitnesses3, cuda_fitnesses4],
                        [cuda_sorted_chromosomes1, cuda_sorted_chromosomes2, cuda_sorted_chromosomes3, cuda_sorted_chromosomes4],
                        [cuda_sorted_fitnesses1, cuda_sorted_fitnesses2, cuda_sorted_fitnesses3, cuda_sorted_fitnesses4],
                        [cuda_fitnessTotal1, cuda_fitnessTotal2, cuda_fitnessTotal3, cuda_fitnessTotal4],
                        [cuda_rouletteWheel1, cuda_rouletteWheel2, cuda_rouletteWheel3, cuda_rouletteWheel4],
                        [cuda_crossoverRate1, cuda_crossoverRate2, cuda_crossoverRate3, cuda_crossoverRate4],
                        [cuda_tmp_fitnesses1, cuda_tmp_fitnesses2, cuda_tmp_fitnesses3, cuda_tmp_fitnesses4],
                        [cuda_mutationRate1, cuda_mutationRate2, cuda_mutationRate3, cuda_mutationRate4],
                    ):
                sort_chromosomes[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_fitnesses,
                                                                     cuda_sorted_chromosomes,
                                                                     cuda_sorted_fitnesses, cuda_fitnessTotal)

                # Crossover And Mutation
                if i != num_generations - 1:
                    crossover[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_sorted_chromosomes, cuda_sorted_fitnesses, cuda_rouletteWheel, states[1], cuda_popSize,
                                                                  cuda_eliteSize, cuda_fitnessTotal,
                                                                  cuda_crossoverRate, cuda_is_adaptive)

                    differential_evolution[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_eliteSize, states[2], states[3], states[4])

                    mutation[blocks_per_grid, threads_per_block](cuda_chromosomes, cuda_eliteSize, cuda_tmp_fitnesses, cuda_N_target, states[5], cuda_mutationRate,
                                                                 cuda_is_adaptive)

            if (i + 1) % 10 == 0:
                information_exchange[blocks_per_grid, threads_per_block](cuda_sorted_chromosomes1, cuda_sorted_chromosomes2, cuda_sorted_chromosomes3, cuda_sorted_chromosomes4,
                                                                         cuda_chromosomes4)


    end = time.perf_counter()
    show_fitness_pairs = []
    chromosomes1 = cuda_chromosomes1.copy_to_host()
    chromosomes2 = cuda_chromosomes2.copy_to_host()
    chromosomes3 = cuda_chromosomes3.copy_to_host()
    chromosomes4 = cuda_chromosomes4.copy_to_host()
    fitnesses1 = cuda_fitnesses1.copy_to_host()
    fitnesses2 = cuda_fitnesses2.copy_to_host()
    fitnesses3 = cuda_fitnesses3.copy_to_host()
    fitnesses4 = cuda_fitnesses4.copy_to_host()

    all_chromosomes = [chromosomes1, chromosomes2, chromosomes3, chromosomes4]
    all_fitnesses = [fitnesses1, fitnesses2, fitnesses3, fitnesses4]

    # 找到最大值及其来源
    max_fitness = float('-inf')  # 初始化为负无穷
    max_group = -1  # 初始化为无效索引
    max_index = -1  # 记录最大值在组内的索引

    for i, fitness_group in enumerate(all_fitnesses):
        group_max = np.max(fitness_group)  # 当前组的最大值
        # print("第{}组".format(i), group_max)
        if group_max >= max_fitness:  # 更新全局最大值
            max_fitness = group_max
            max_group = i + 1  # 记录组编号（从1开始）
            max_index = np.argmax(fitness_group)  # 组内索引

    # 输出结果
    # print(f"max fitness: {max_fitness}")
    # print(f"from:  {max_group} th group,  {max_index} th solution")

    best_chromosomes = all_chromosomes[max_group - 1]
    best_fitnesses = all_fitnesses[max_group - 1]

    for i in range(len(best_chromosomes)):
        show_fitness_pairs.append([best_chromosomes[i], best_fitnesses[i]])
    fitnesses = list(reversed(sorted(best_fitnesses)))  # fitnesses now in descending order
    show_sorted_pairs = list(reversed(sorted(show_fitness_pairs, key=lambda x: x[1])))
    best_allocation = show_sorted_pairs[0][0]
    best_fitness = show_sorted_pairs[0][1]
    return best_allocation, best_fitness[0], end - start
