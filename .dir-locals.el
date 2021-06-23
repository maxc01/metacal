;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((python-mode
  (eval add-to-list 'python-shell-extra-pythonpaths (cdr (project-current)))
  (eval venv-workon "metacal")
  (eval message "Now in dp virtualenv")))
