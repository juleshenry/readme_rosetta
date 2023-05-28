import sys
if __name__=='__main__':
	sa = sys.argv
	if not len(sa) == 4:
		error = "Must have form ~`babel aa bb target.file`"
		raise ValueError(error)
	print(f"Converting {argv[1]} => {argv[2]} on {argv[3]}")
	