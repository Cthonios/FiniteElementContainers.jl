function check_color(colors, color_counts, elem_to_elem, e, n_threads)

	temp_color = 1
	for n in elem_to_elem

		# skip identity
		if n == e continue end

		# check for neighbor color
		if colors[n] == temp_color
			temp_color = temp_color + 1
		end

		# check for max n_threads
		if length(color_counts) >= temp_color
			if color_counts[temp_color] == n_threads
				temp_color = temp_color + 1
			end
		end
	end

	return temp_color
end

function greedy_element_coloring_mine(file_name::String, n_threads::Int)

	exo = ExodusDatabase(file_name, "r")
	elem_to_elem = collect_element_to_element_connectivities(exo)
	close(exo)
	copy_mesh(file_name, file_name * ".col")

	colors = zeros(Int, length(elem_to_elem))
	colors[1] = 1

	color_counts = [1]

	for e in 2:length(elem_to_elem)
		temp_color = check_color(colors, color_counts, elem_to_elem[e], e, n_threads)

		if length(color_counts) >= temp_color
			color_counts[temp_color] = color_counts[temp_color] + 1
		else
			push!(color_counts, 1)
		end
		# display(color_counts)

		colors[e] = temp_color

	end

	exo = ExodusDatabase(file_name * ".col", "rw")
	write_names(exo, ElementVariable, ["color"])
	write_time(exo, 1, 0.0)
	write_values(exo, ElementVariable, 1, 1, "color", convert.(Float64, colors))
	close(exo)
	return colors

	end

function greedy_element_coloring(file_name::String, n_threads::Int)
	exo = ExodusDatabase(file_name, "r")
	elem_to_elem = collect_element_to_element_connectivities(exo)
	close(exo)
	copy_mesh(file_name, file_name * ".col")


	# algorithm below
	cell_colors = Dict{Int, Int}(i => 0 for i in 1:length(elem_to_elem))
	occupied_colors = Set{Int}()
	final_colors = Vector{Int}[]
	color_counts = Int[]
	total_colors = 0

	for e in axes(elem_to_elem, 1)
		empty!(occupied_colors)

		# look at neighbors
		for n in elem_to_elem[e]
			# ignore identity
			if n == e continue end

			color = cell_colors[n]
			if color != 0
				push!(occupied_colors, color)
			end

		end

		# occupied colors now contains all teh colors we are not allowed to use
		free_color = 0
		for attempt_color in 1:total_colors
			if attempt_color ∉ occupied_colors
				free_color = attempt_color
				if color_counts[free_color] != n_threads
					break
				else
					free_color = 0
				end
			end
		end

		if free_color == 0
			total_colors = total_colors + 1
			free_color = total_colors
			push!(final_colors, Int[])
			push!(color_counts, 0)
		end

		@assert free_color != 0
		cell_colors[e] = free_color
		push!(final_colors[free_color], e)
		color_counts[free_color] += 1
	end

	colors = Vector{Int}(undef, length(keys(cell_colors)))

	for (key, val) in cell_colors
		colors[key] = val
	end

	# now check colors
	for (e, color) in enumerate(colors)
		for n in elem_to_elem[e]
			if n == e continue end
			# @show color
			if colors[n] == color
				println("Error!")
			end
		end
	end

	# display(colors)
	exo = ExodusDatabase(file_name * ".col", "rw")
	write_names(exo, ElementVariable, ["color"])
	write_time(exo, 1, 0.0)

	display(color_counts)
	n_els = 1
	for block in read_sets(exo, Block)
		temp_colors = convert.(Float64, colors[n_els:n_els - 1 + size(block.conn, 2)])
		write_values(exo, ElementVariable, 1, block.id, "color", temp_colors)
		n_els = n_els + size(block.conn, 2)
	end
	close(exo)
end