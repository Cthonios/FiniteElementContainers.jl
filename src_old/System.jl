struct System{Rtype, I, B, NDof}
	mesh::Mesh{Rtype, I, B}
	dof::DofManager{NDof}
	fspaces
	asm
	bcs
end

function System()

end