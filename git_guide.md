# Comprehensive Git Guide with Visual Diagrams

This guide provides visual representations of Git concepts, workflows, and operations using Mermaid diagrams.

## Table of Contents

1. [Git Basics](#git-basics)
2. [Git Areas and Workflow](#git-areas-and-workflow)
3. [Branching and Merging](#branching-and-merging)
4. [Remote Operations](#remote-operations)
5. [Collaboration Workflows](#collaboration-workflows)
6. [Advanced Operations](#advanced-operations)
7. [Git History and Navigation](#git-history-and-navigation)

---

## Git Basics

### Git Repository Structure

```mermaid
graph TD
    A[Git Repository] --> B[Working Directory]
    A --> C[Staging Area/Index]
    A --> D[Local Repository]
    A --> E[Remote Repository]

    B --> B1[Modified Files]
    B --> B2[Untracked Files]
    B --> B3[Tracked Files]

    C --> C1[Staged Changes]
    C --> C2[Ready for Commit]

    D --> D1[Commit History]
    D --> D2[Branches]
    D --> D3[Tags]

    E --> E1[Origin]
    E --> E2[Upstream]
    E --> E3[Other Remotes]
```

---

## Git Areas and Workflow

### Basic Git Workflow

```mermaid
graph LR
    A[Working Directory] -->|git add| B[Staging Area]
    B -->|git commit| C[Local Repository]
    C -->|git push| D[Remote Repository]
    D -->|git fetch| E[Remote Tracking Branch]
    E -->|git merge| A
    D -->|git pull| A

    style A fill:#ff9999
    style B fill:#ffcc99
    style C fill:#99ccff
    style D fill:#99ff99
```

### File States and Transitions

```mermaid
stateDiagram-v2
    [*] --> Untracked: Create new file
    Untracked --> Staged: git add
    Staged --> Unmodified: git commit
    Unmodified --> Modified: Edit file
    Modified --> Staged: git add
    Staged --> Modified: Edit staged file
    Modified --> Unmodified: git checkout
    Unmodified --> Untracked: git rm
    Staged --> Untracked: git rm --cached
```

### Complete Git Command Flow

```mermaid
flowchart TD
    Start[Start Working] --> Check{Check Status}
    Check -->|git status| Modified{Files Modified?}

    Modified -->|Yes| Stage[git add files]
    Modified -->|No| Pull[git pull]

    Stage --> Review[Review Changes]
    Review -->|git diff --staged| Commit[git commit -m message]

    Commit --> Push{Push to Remote?}
    Push -->|Yes| PushCmd[git push origin branch]
    Push -->|No| Continue[Continue Working]

    PushCmd --> Success{Success?}
    Success -->|Yes| Done[Done]
    Success -->|No| Resolve[Resolve Conflicts]

    Resolve --> Pull
    Pull --> Merge{Conflicts?}
    Merge -->|Yes| ResolveConflict[Resolve Conflicts]
    Merge -->|No| Continue

    ResolveConflict --> Stage
    Continue --> Start
```

---

## Branching and Merging

### Git Branching Model

```mermaid
gitGraph
    commit id: "Initial commit"
    commit id: "Add README"

    branch develop
    checkout develop
    commit id: "Setup project structure"
    commit id: "Add configuration"

    branch feature/user-auth
    checkout feature/user-auth
    commit id: "Add login form"
    commit id: "Add authentication logic"
    commit id: "Add tests"

    checkout develop
    merge feature/user-auth
    commit id: "Integrate user auth"

    checkout main
    merge develop tag: "v1.0.0"

    checkout develop
    branch feature/dashboard
    checkout feature/dashboard
    commit id: "Create dashboard"
    commit id: "Add widgets"

    checkout develop
    branch bugfix/login-error
    checkout bugfix/login-error
    commit id: "Fix login bug"

    checkout develop
    merge bugfix/login-error
    merge feature/dashboard

    checkout main
    merge develop tag: "v1.1.0"
```

### Branch Operations

```mermaid
graph TD
    A[main branch] -->|git branch feature| B[Create feature branch]
    B -->|git checkout feature| C[Switch to feature]
    C -->|Make commits| D[feature branch]
    D -->|git checkout main| E[Switch to main]
    E -->|git merge feature| F[Merge feature into main]
    F -->|git branch -d feature| G[Delete feature branch]

    A -->|git checkout -b hotfix| H[Create & switch to hotfix]
    H -->|Make commits| I[hotfix branch]
    I -->|git checkout main| J[Switch to main]
    J -->|git merge hotfix| K[Merge hotfix]

    style A fill:#99ff99
    style D fill:#ffcc99
    style F fill:#99ccff
    style I fill:#ff9999
```

### Merge vs Rebase

```mermaid
graph TD
    subgraph "Merge Strategy"
        M1[main: A-B-C] --> M2[feature: A-B-D-E]
        M1 --> M3[main after merge: A-B-C-M]
        M2 --> M3
        M3 --> M4[Merge commit M combines C and E]
    end

    subgraph "Rebase Strategy"
        R1[main: A-B-C] --> R2[feature: A-B-D-E]
        R1 --> R3[feature after rebase: A-B-C-D'-E']
        R3 --> R4[Linear history, no merge commit]
    end

    style M3 fill:#ffcc99
    style M4 fill:#ff9999
    style R3 fill:#99ccff
    style R4 fill:#99ff99
```

### Conflict Resolution Flow

```mermaid
flowchart TD
    A[Start Merge/Rebase] --> B{Conflicts?}
    B -->|No| C[Merge Complete]
    B -->|Yes| D[Git marks conflicts]

    D --> E[Open conflicted files]
    E --> F[Find conflict markers]
    F --> G["<<<<<<< HEAD<br/>your changes<br/>=======<br/>their changes<br/>>>>>>>> branch"]

    G --> H[Choose resolution]
    H --> I[Edit file to resolve]
    I --> J[Remove conflict markers]
    J --> K[Save file]

    K --> L[git add resolved-file]
    L --> M{More conflicts?}
    M -->|Yes| E
    M -->|No| N[git commit or git rebase --continue]
    N --> C

    style G fill:#ff9999
    style K fill:#ffcc99
    style C fill:#99ff99
```

---

## Remote Operations

### Remote Repository Operations

```mermaid
graph TD
    A[Local Repository] -->|git push| B[Remote Repository]
    B -->|git fetch| C[Remote Tracking Branches]
    C -->|git merge| A
    B -->|git pull = fetch + merge| A

    A -->|git push origin branch| D[Create remote branch]
    D -->|git push -u origin branch| E[Set upstream tracking]

    B -->|git clone| F[New Local Repository]
    F -->|git remote add| G[Add additional remotes]

    A -->|git push --tags| H[Push tags to remote]
    B -->|git fetch --tags| I[Fetch tags from remote]

    style A fill:#99ccff
    style B fill:#99ff99
    style C fill:#ffcc99
```

### Fork and Pull Request Workflow

```mermaid
sequenceDiagram
    participant O as Original Repo
    participant F as Your Fork
    participant L as Local Clone
    participant PR as Pull Request

    O->>F: Fork repository
    F->>L: git clone
    L->>L: git checkout -b feature
    L->>L: Make changes & commit
    L->>F: git push origin feature
    F->>PR: Create Pull Request
    PR->>O: Request to merge
    O->>O: Code review
    O->>O: Merge PR
    O->>L: git pull upstream main
    L->>L: git checkout main
    L->>L: git merge upstream/main
    L->>F: git push origin main
```

### Multi-Remote Setup

```mermaid
graph TD
    A[Local Repository] --> B[origin remote]
    A --> C[upstream remote]
    A --> D[deploy remote]

    B --> B1[Your fork on GitHub]
    C --> C1[Original project repo]
    D --> D1[Production server]

    A -->|git push origin| B
    A -->|git pull upstream| C
    A -->|git push deploy main| D

    C1 -->|git fetch upstream| A
    A -->|git merge upstream/main| A

    style A fill:#99ccff
    style B1 fill:#ffcc99
    style C1 fill:#99ff99
    style D1 fill:#ff9999
```

---

## Collaboration Workflows

### Gitflow Workflow

```mermaid
graph TD
    A[Production: main] --> B[Development: develop]
    B --> C[Feature Branches]
    B --> D[Release Branches]
    A --> E[Hotfix Branches]

    C --> C1[feature/login]
    C --> C2[feature/dashboard]
    C --> C3[feature/api]

    C1 -->|Merge when complete| B
    C2 -->|Merge when complete| B
    C3 -->|Merge when complete| B

    B -->|Ready for release| D
    D --> D1[release/1.0]
    D1 -->|Bug fixes only| D1
    D1 -->|Final testing| A
    D1 -->|Merge back| B

    A -->|Critical bug| E
    E --> E1[hotfix/security-patch]
    E1 -->|Merge when fixed| A
    E1 -->|Merge back| B

    style A fill:#ff9999
    style B fill:#99ccff
    style C fill:#ffcc99
    style E fill:#ff99cc
```

### Feature Branch Workflow

```mermaid
sequenceDiagram
    participant M as main
    participant F as feature/new-feature
    participant Dev as Developer

    M->>F: Create branch
    Dev->>F: Work on feature
    F->>F: Multiple commits
    Dev->>F: git commit -m "Add feature part 1"
    Dev->>F: git commit -m "Add feature part 2"
    Dev->>F: git commit -m "Add tests"

    M->>M: Other changes merged
    F->>F: git fetch origin main
    F->>F: git rebase main (or merge)
    F->>F: Resolve conflicts if any

    F->>M: Create Pull Request
    M->>M: Code Review
    M->>M: CI/CD checks
    M->>M: Merge feature
    F->>F: Delete branch
```

### Team Collaboration Flow

```mermaid
flowchart TD
    A[Team Member 1] -->|Push to feature branch| B[Shared Repository]
    C[Team Member 2] -->|Push to different feature| B
    D[Team Member 3] -->|Push to another feature| B

    B -->|Pull Request| E{Code Review}
    E -->|Approved| F[Merge to develop]
    E -->|Changes Requested| G[Update code]
    G -->|Push updates| B

    F --> H[CI/CD Pipeline]
    H -->|Tests Pass| I[Deploy to Staging]
    H -->|Tests Fail| J[Fix Issues]
    J --> G

    I -->|QA Approval| K[Merge to main]
    K --> L[Deploy to Production]

    B -->|Daily| M[git pull origin develop]
    M --> A
    M --> C
    M --> D

    style E fill:#ffcc99
    style F fill:#99ff99
    style H fill:#99ccff
    style L fill:#ff9999
```

---

## Advanced Operations

### Cherry-Pick Operation

```mermaid
gitGraph
    commit id: "A"
    commit id: "B"
    branch feature
    checkout feature
    commit id: "C"
    commit id: "D" type: HIGHLIGHT
    commit id: "E"
    checkout main
    commit id: "F"
    cherry-pick id: "D"
    commit id: "G"
```

### Rebase Interactive Flow

```mermaid
flowchart TD
    A[Start: git rebase -i HEAD~5] --> B[Editor opens with commits]
    B --> C{Choose action for each commit}

    C -->|pick| D[Keep commit as is]
    C -->|reword| E[Change commit message]
    C -->|edit| F[Pause to modify commit]
    C -->|squash| G[Combine with previous commit]
    C -->|fixup| H[Combine, discard message]
    C -->|drop| I[Remove commit]

    D --> J[Save and close editor]
    E --> J
    F --> K[Make changes]
    K --> L[git commit --amend]
    L --> M[git rebase --continue]
    G --> J
    H --> J
    I --> J

    J --> N{Conflicts?}
    N -->|Yes| O[Resolve conflicts]
    O --> P[git add resolved-files]
    P --> M
    N -->|No| Q[Rebase complete]
    M --> Q

    style C fill:#ffcc99
    style Q fill:#99ff99
```

### Stash Operations

```mermaid
graph TD
    A[Working Directory<br/>with changes] -->|git stash| B[Changes stashed<br/>Clean working dir]
    B -->|git stash list| C[View stash list]
    B -->|git stash pop| D[Apply & remove stash]
    B -->|git stash apply| E[Apply & keep stash]
    B -->|git stash drop| F[Remove stash]

    A -->|git stash save message| G[Stash with description]
    G --> C

    B -->|git stash show| H[Preview stash contents]
    B -->|git stash branch| I[Create branch from stash]

    style A fill:#ff9999
    style B fill:#99ccff
    style D fill:#99ff99
```

### Reset vs Revert

```mermaid
graph TD
    subgraph "git reset"
        A1[HEAD at commit C] -->|git reset --soft B| A2[HEAD at B<br/>Changes staged]
        A1 -->|git reset --mixed B| A3[HEAD at B<br/>Changes unstaged]
        A1 -->|git reset --hard B| A4[HEAD at B<br/>Changes discarded]
    end

    subgraph "git revert"
        B1[Commits: A-B-C] -->|git revert C| B2[Commits: A-B-C-C']
        B2 --> B3[C' undoes changes from C]
    end

    style A2 fill:#99ff99
    style A3 fill:#ffcc99
    style A4 fill:#ff9999
    style B2 fill:#99ccff
```

### Tag Management

```mermaid
graph TD
    A[Commit History] -->|git tag v1.0.0| B[Lightweight Tag]
    A -->|git tag -a v1.0.0 -m msg| C[Annotated Tag]

    C --> D[Contains:<br/>- Tagger name<br/>- Email<br/>- Date<br/>- Message]

    B -->|git push origin v1.0.0| E[Push single tag]
    C -->|git push --tags| F[Push all tags]

    A -->|git tag -l v1.*| G[List matching tags]
    A -->|git show v1.0.0| H[Show tag details]

    B -->|git tag -d v1.0.0| I[Delete local tag]
    E -->|git push origin :v1.0.0| J[Delete remote tag]

    style C fill:#99ccff
    style D fill:#ffcc99
```

---

## Git History and Navigation

### Commit History Visualization

```mermaid
gitGraph
    commit id: "Initial commit" tag: "v0.1.0"
    commit id: "Add README"
    branch develop
    checkout develop
    commit id: "Project setup"

    branch feature/auth
    checkout feature/auth
    commit id: "Add login"
    commit id: "Add registration"

    checkout develop
    branch feature/ui
    checkout feature/ui
    commit id: "Add navbar"
    commit id: "Add footer"

    checkout develop
    merge feature/auth

    checkout main
    merge develop tag: "v0.2.0"

    checkout develop
    merge feature/ui
    commit id: "Integration tests"

    checkout main
    merge develop tag: "v0.3.0"

    checkout main
    branch hotfix/security
    checkout hotfix/security
    commit id: "Security patch"

    checkout main
    merge hotfix/security tag: "v0.3.1"

    checkout develop
    merge hotfix/security
```

### Navigating History

```mermaid
graph TD
    A[HEAD] --> B[Current commit]
    B --> C[HEAD~1 or HEAD^]
    C --> D[HEAD~2 or HEAD^^]
    D --> E[HEAD~3 or HEAD^^^]

    B --> F[branch-name]
    B --> G[commit-hash]
    B --> H[tag-name]

    A -->|git checkout HEAD~2| I[Move to 2 commits back]
    A -->|git checkout branch| J[Move to branch tip]
    A -->|git checkout commit-hash| K[Move to specific commit]

    B -->|git log| L[View commit history]
    B -->|git log --graph| M[View graphical history]
    B -->|git log --oneline| N[View compact history]
    B -->|git reflog| O[View reference log]

    style A fill:#ff9999
    style B fill:#99ccff
    style L fill:#99ff99
```

### Commit Graph Relationships

```mermaid
graph TD
    A[Commit A<br/>Initial] --> B[Commit B<br/>Add feature]
    B --> C[Commit C<br/>Fix bug]
    B --> D[Commit D<br/>Branch feature]
    D --> E[Commit E<br/>Develop feature]
    C --> F[Commit F<br/>Merge commit]
    E --> F
    F --> G[Commit G<br/>Latest main]

    H[HEAD] -.-> G
    I[main] -.-> G
    J[origin/main] -.-> F
    K[feature/branch] -.-> E

    style A fill:#e1f5e1
    style G fill:#99ff99
    style F fill:#ffcc99
    style H fill:#ff9999
```

### Understanding Git References

```mermaid
graph LR
    A[Git References] --> B[Branches]
    A --> C[Tags]
    A --> D[HEAD]
    A --> E[Remote References]

    B --> B1[refs/heads/main]
    B --> B2[refs/heads/develop]
    B --> B3[refs/heads/feature/*]

    C --> C1[refs/tags/v1.0.0]
    C --> C2[refs/tags/v2.0.0]

    D --> D1[Current branch pointer]
    D --> D2[Detached HEAD state]

    E --> E1[refs/remotes/origin/main]
    E --> E2[refs/remotes/upstream/main]

    style A fill:#99ccff
    style B fill:#99ff99
    style C fill:#ffcc99
    style D fill:#ff9999
    style E fill:#cc99ff
```

---

## Best Practices

### Commit Message Convention

```mermaid
graph TD
    A[Good Commit Message] --> B[Type: feat, fix, docs, style, refactor, test, chore]
    A --> C[Scope: component or file affected]
    A --> D[Subject: short description]
    A --> E[Body: detailed explanation optional]
    A --> F[Footer: breaking changes, issue references]

    G[Example] --> H["feat(auth): add OAuth2 login<br/><br/>Implement OAuth2 authentication flow<br/>- Add login endpoint<br/>- Add token validation<br/>- Add refresh token logic<br/><br/>Closes #123"]

    style A fill:#99ccff
    style G fill:#99ff99
    style H fill:#ffcc99
```

### Branch Naming Convention

```mermaid
graph TD
    A[Branch Name] --> B[Type/Description]

    B --> C[feature/user-authentication]
    B --> D[bugfix/login-error]
    B --> E[hotfix/security-patch]
    B --> F[release/v1.2.0]
    B --> G[docs/api-documentation]
    B --> H[refactor/database-layer]
    B --> I[test/integration-tests]

    style A fill:#99ccff
    style C fill:#99ff99
    style D fill:#ffcc99
    style E fill:#ff9999
```

---

## Troubleshooting

### Common Issues and Solutions

```mermaid
flowchart TD
    A[Git Issue] --> B{What's wrong?}

    B -->|Committed to wrong branch| C[git cherry-pick commit-hash<br/>to correct branch]
    B -->|Need to undo last commit| D[git reset --soft HEAD~1]
    B -->|Pushed sensitive data| E[git filter-branch or BFG]
    B -->|Merge conflicts| F[Resolve manually,<br/>git add, git commit]
    B -->|Lost commits| G[git reflog to find,<br/>git cherry-pick to restore]
    B -->|Detached HEAD| H[git checkout branch-name<br/>or git switch -c new-branch]

    C --> Z[Resolved]
    D --> Z
    E --> Z
    F --> Z
    G --> Z
    H --> Z

    style B fill:#ffcc99
    style Z fill:#99ff99
```

---

## Quick Reference

### Essential Git Commands

| Command | Description | Diagram Reference |
|---------|-------------|-------------------|
| `git init` | Initialize a new repository | Repository Structure |
| `git clone <url>` | Clone a repository | Remote Operations |
| `git add <file>` | Stage changes | Basic Workflow |
| `git commit -m "msg"` | Commit staged changes | Basic Workflow |
| `git status` | Check working directory status | File States |
| `git log` | View commit history | Navigating History |
| `git branch <name>` | Create a new branch | Branch Operations |
| `git checkout <branch>` | Switch branches | Branch Operations |
| `git merge <branch>` | Merge branch into current | Branching Model |
| `git pull` | Fetch and merge from remote | Remote Operations |
| `git push` | Push commits to remote | Remote Operations |
| `git stash` | Temporarily save changes | Stash Operations |
| `git rebase <branch>` | Reapply commits on top of another | Rebase Flow |
| `git cherry-pick <hash>` | Apply specific commit | Cherry-Pick |
| `git reset <mode> <ref>` | Reset HEAD to a previous state | Reset vs Revert |
| `git revert <commit>` | Create new commit undoing changes | Reset vs Revert |
| `git tag <name>` | Create a tag | Tag Management |

---

## Additional Resources

- [Official Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Interactive Git Branching Tutorial](https://learngitbranching.js.org/)

---

*This guide is designed to help developers understand Git through visual diagrams. Each diagram can be rendered using Mermaid in Markdown-compatible viewers.*
