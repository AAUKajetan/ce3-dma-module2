.PHONY: check check0 check1 check2 check3 check4 demo solutions clean

check: ## check 0 verifies the datasets, then the four labs; a failure names the lab and the fix
	@python3 verify/check_00_data.py; s0=$$?; \
	 python3 verify/check_01.py; s1=$$?; \
	 python3 verify/check_02.py; s2=$$?; \
	 python3 verify/check_03.py; s3=$$?; \
	 python3 verify/check_04.py; s4=$$?; \
	 echo; echo "data: $$s0   lab 1: $$s1   lab 2: $$s2   lab 3: $$s3   lab 4: $$s4"; \
	 echo "0 = green   1 = written, not right yet   2 = not written yet   3 = environment not ready, run  bash setup.sh"

check0: ; @python3 verify/check_00_data.py
check1: ; @python3 verify/check_01.py
check2: ; @python3 verify/check_02.py
check3: ; @python3 verify/check_03.py
check4: ; @python3 verify/check_04.py

demo: ## run the four reference solutions with narration; figures land in out/, index in out/index.html
	@for k in 1 2 3 4; do echo; echo "=============== solution $$k ==============="; \
	   python3 solutions/lab_0$$k.py || echo "solution $$k exited $$?"; done; \
	 python3 -c "import _narrate; _narrate.demo_index()"

solutions: ## copy the shipped solutions over the labs, to read or to recover
	@python3 apply.py

clean: ; @rm -rf landing __pycache__ */__pycache__ DATA_PROFILE.md out timing.json \
	          labs/.attempts.json
# labs/.attempts.json is the checks' count of how many times each assertion has
# failed, which is all that decides when a message stops being a hint and starts
# being an explanation. Removing it resets the hints, which is a fair thing for
# `clean` to do. labs/.your_attempt is NOT removed here -- see `reset` below.

# `clean` used to delete labs/.your_attempt as well, which is where apply.py
# puts YOUR file before it copies a solution over it. Write an answer, peek at
# the solution, tidy up, and your answer was gone -- and `--restore` then said
# nothing and exited 0. Removing the backup is now its own target, and it says
# what it is about to destroy.
reset: ; @echo "This deletes labs/.your_attempt -- the copy of YOUR work that"; \
	@echo "apply.py saved before overwriting it. Ctrl-C now if you want it."; \
	@echo; read -p "type yes to delete it: " answer; \
	[ "$$answer" = yes ] && rm -rf labs/.your_attempt && echo "deleted" || echo "kept"

