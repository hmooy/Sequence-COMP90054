git push --delete origin submission-contest
git tag --delete submission-contest
git checkout main
git tag -a submission-contest -m  "fix urgent place"
git push --tags