
dry_baroclinic_sphere:
	mpiexec -n 16 \
		python3 dry_baroclinic_sphere.py \
			--ncells_per_edge=16 \
			--nlayers=15 \
			--dt=900.0 \
			--tmax=1296000.0 \
			--dumpfreq=48

moist_bryan_fritsch:
	mpiexec -n 4 \
		python3 moist_bryan_fritsch.py \
			--ncolumns=50 \
			--nlayers=50 \
			--dt=2.0 \
			--tmax=1000.0
			--dumpfreq=15

