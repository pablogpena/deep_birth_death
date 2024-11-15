################################################################
# LTT plot for multifurcation tree t
ltt.plot.gen <- function (t,color="black",width=1) {
	ltt <- ltt.general(t)
	#plot(ltt,type="l",log = "y",xlab = "time", ylab = "number of species")
	plot(ltt,type="l",xlab = "Time before present", ylab = "Number of species", log="y",col=color,xaxt="n",lwd=width)
	axis(1,at=50*(0:-3),labels=50*(0:3),las=1)
	}	

################################################################
# LTT plot for multifurcation tree t
ltt.plot.gen.2 <- function (t,color="black",width=1) {
	ltt <- ltt.general(t)
	#plot(ltt,type="l",log = "y",xlab = "time", ylab = "number of species")
	plot(c(-60,0),c(2,350),type="l",xlab = "Time before present", ylab = "Number of species", log="y",col="white",xaxt="n",lwd=width)
	lines(ltt,col=color)
	axis(1,at=10*(0:-6),labels=10*(0:6),las=1)
	ltt
	}	

################################################################
# Add line to LTT plot for multifurcation tree t
ltt.lines.gen.2 <- function (t,color,width=1) {
	ltt <- ltt.general(t)
	lines(ltt,type="l",col=color,lwd=width)
	ltt
	}

################################################################
# Add line to LTT plot for multifurcation tree t
ltt.lines.gen <- function (t,color,width=1) {
	ltt <- ltt.general(t)
	lines(ltt,type="l",col=color,lwd=width)
	}	

################################################################
# calculates joint LTT plot of m trees (ancestral lineage is omitted in all trees - ie. final number of species is by m lower if m trees)
mltt.general <- function (trees) { 
furcation <- vector()
branchingtree <- vector()
notultra <- vector()
for (j in 1:length(trees)){
	if (is.ultrametric(trees[[j]]) == TRUE) {
		branching <- ltt.general(trees[[j]])
		branchingdiff <- branching[1,2]-1
		for (k in 2:(length(branching[,1])-1)){
			branchingdiff <- c(branchingdiff,(branching[k,2]-branching[(k-1),2]))
			}
		furcation <- c(furcation, branchingdiff)
		branchingtemp <- branching[,1]
		branchingtemp <- branchingtemp[-length(branchingtemp)]
		branchingtree <- c(branchingtree, branchingtemp)

	} else {
			notultra <- c(notultra, j)
			}
}
furcation <- furcation[order(branchingtree)]
current <- furcation[1]
linnumber <- current
for (j in 2:(length(furcation))){
	current <- current + furcation[j]
	linnumber <- c(linnumber, current)
	}
branchingtree <- sort(branchingtree)
branchingtree <- c(branchingtree,0)
linnumber <- c(linnumber,linnumber[length(linnumber)])
obj<-cbind(branchingtree, linnumber)
numbevents <- length(obj[,1])
for (j in 1: (numbevents-2)) {
	if (obj[numbevents-j,1] == obj[numbevents-(j+1),1]){
		obj <- obj[-(numbevents-(j+1)),]
		}
	}
obj
}


################################################################
# calculates avg LTT plot of m reconstructed trees (ancestral lineage is NOT omitted in the trees)
mltt.avg <- function (trees) { 
furcation <- vector()
branchingtree <- vector()
notultra <- vector()
for (j in 1:length(trees)){
	if (is.ultrametric(trees[[j]]) == TRUE) {
		branching <- ltt.general(trees[[j]])
		branchingdiff <- branching[1,2]
		for (k in 2:(length(branching[,1])-1)){
			branchingdiff <- c(branchingdiff,(branching[k,2]-branching[(k-1),2]))
			}
		furcation <- c(furcation, branchingdiff)
		branchingtemp <- branching[,1]
		branchingtemp <- branchingtemp[-length(branchingtemp)]
		branchingtree <- c(branchingtree, branchingtemp)
    print(paste("Ultrametrico procesado", j))
    flush.console()
	} else {
			notultra <- c(notultra, j)
			}
}
furcation <- furcation[order(branchingtree)]
current <- furcation[1]
linnumber <- current
for (j in 2:(length(furcation))){
	current <- current + furcation[j]
	linnumber <- c(linnumber, current)
	}
branchingtree <- sort(branchingtree)
branchingtree <- c(branchingtree,0)
linnumber <- c(linnumber,linnumber[length(linnumber)])
obj<-cbind(branchingtree, linnumber)
numbevents <- length(obj[,1])
for (j in 1: (numbevents-2)) {
    print(paste("numbevents", j))
    flush.console()
	if (obj[numbevents-j,1] == obj[numbevents-(j+1),1]){
		obj <- obj[-(numbevents-(j+1)),]
		}
	}
for (j in 1:length(obj[,2])){
	obj[j,2]<- obj[j,2]/length(trees)
    print(paste("obj", j))
    flush.console()
	}
obj
}


mltt.avg.reconst <- function(trees) {
	treesrec<-list()
	for (j in 1:length(trees)){
		treesrec <- c(treesrec,list(drop.extinct(trees[[j]],tol = 0.000001)))
        print(paste("pruneados", j))
        flush.console()
		}
	v<-mltt.avg(treesrec)
	v
	}

## calculates avg LTT (complete+reconstructed) plot of m complete trees (ancestral lineage is NOT omitted in the trees)
mltt.avg.complete <- function(trees) {
	treesrec <- list()
	plotcomp <- vector()
	for (j in 1:length(trees)){
		#treesrec <- c(treesrec,list(prune.extinct.taxa(trees[[j]],tol = 0.000001)))
		compl <- getnumbs(trees[[j]])
		for (k in length(compl[,3]):2){
			compl[k,3] <- compl[k,3]-compl[(k-1),3]
			}
		compl <- compl[-length(compl[,1]),]
		plotcomp<-rbind(plotcomp,compl)
        print(paste("procesados", j))
        flush.console()
		}
	ycomp <- plotcomp[,3]
	xcomp <- - plotcomp[,2]
	ycomp <- ycomp[order(xcomp)]
	xcomp <- sort(xcomp)
	for (j in 2:length(xcomp)){
		ycomp[j]<-ycomp[j-1]+ycomp[j]
        print(paste("xcomp", j))
        flush.console()
		}
	for (j in 1:length(ycomp)){
		ycomp[j]<-ycomp[j]/length(trees)
        print(paste("ycomp", j))
        flush.console()
		}
	#v<-mltt.avg(treesrec)
	
	
	
	#plot(v,type='l',ylog=TRUE)

	
	plotcomp<-cbind(xcomp,ycomp)
	plotcomp
	}


## plots ltt for complete tree
plot.ltt.complete <- function(tree) {
	compl <- getnumbs(tree)
		for (k in length(compl[,3]):2){
			compl[k,3] <- compl[k,3]-compl[(k-1),3]
            print(paste("compl", k))
            flush.console()
			}
		plotcomp <- compl[-length(compl[,1]),]
	ycomp <- plotcomp[,3]
	xcomp <- - plotcomp[,2]
	ycomp <- ycomp[order(xcomp)]
	xcomp <- sort(xcomp)
	for (j in 2:length(xcomp)){
		ycomp[j]<-ycomp[j-1]+ycomp[j]
        print(paste("comp", j))
        flush.console()
		}
	plot(xcomp,ycomp,type='l')
	}


# #########################################
# #
# START FUNCTIONS

################################################################
# gives the time of first split with number of species descending from the first ancesotr
origin <- function(trees){
	origintime <- vector()
	originspecies <- vector()
	for (j in 1:length(trees)) {
		if (is.ultrametric(trees[[j]]) == TRUE) {
			branching <- ltt.general(trees[[j]])
			origintime <- c(origintime, branching[1,1])
			originspecies <- c(originspecies, (branching[1,2]-1))
			}
		}
		origintree <- cbind(origintime,originspecies)
		origintree
}

################################################################
# gives the time of first split with number of species descending from the first ancesotr
origininLTT <- function(origin,lttplot){
	originspecies <- vector()
	for (j in 1:length(origin[,1])) {
		for (k in 1:length(lttplot[,1])) {
			if (origin[j,1] == lttplot[k,1])
				originspecies <- c(originspecies,lttplot[k,2])			}
		}
		origintree <- cbind(origin[,1],originspecies)
		origintree
}

################################################################
# gives list of the trees which are NOT ultrametric
isnot.ultrametric <- function(trees){
	notultra <- vector()
	for (j in 1:length(trees)){
		if (is.ultrametric(trees[[j]]) == FALSE) {			notultra <- c(notultra, j)
			}
	}
	notultra
}




####FROM TREESIM
#input: age of extinct species. branching times.
#returns: left column node number, middle column time of node, right column number of species after event
numbspecies <- function(tipdepth,branching){
	branchingorder <- order(branching, decreasing=TRUE)
	branchingdepth<-sort(branching,decreasing=TRUE)
	tiporder<-order(tipdepth[,2],decreasing=TRUE)
	tipdepth<-tipdepth[,2][order(tipdepth[,2],decreasing=TRUE)]
	br<-1
	ext<-1
	numbscur <- 1
	event <- vector()
	numbs <- vector()		#numbs - left column node name, middle column time of node, right column number of species after event
	while (tipdepth[ext] > 0 || br <= length(branching)){  # (ext <= length(tipdepth) || br <= length(branching)){        #
		if (br <= length(branching) && branchingdepth[br] > tipdepth[ext]) {
			numbscur <- numbscur + 1
			event <- c(branchingorder[br]+length(tipdepth), branchingdepth[br], numbscur)
			br <- br+1
		} else {
			numbscur <- numbscur - 1
			event <- c(tiporder[ext], tipdepth[ext], numbscur)
			ext <- ext + 1
		}
		numbs <- rbind(numbs,event)
	}
	numbs
	}	


getnumbs <- function(tree){
		tipdepth <- age.tips(tree)		
		# entry i is the age of tip i
		branching <- branching.times.complete(tree)
		numbs <- numbspecies(tipdepth,branching)
		 
		#numbs - left column node name, middle column time of node, right column number of species after event
		event <- c(0, 0, 0)   #for last interval in next for loop being ok
		numbs <- rbind(numbs,event)
		numbs
		}

#calculates age (i.e. time since death) of extinct species
age.tips <- function(tree){	
	times <- vector()
	if (class(tree)=="phylo"){
	tipdepth <- vector()
	for (j in (1:length(tree$tip))) {
		parent <- tree$edge[,1][tree$edge[,2]==j]
		tipdepthtemp <- branching.times.complete(tree)[as.character(parent)]
		tipdepthtemp <- c(j,round(tipdepthtemp - tree$edge.length[tree$edge[,2]==j],10))
		tipdepth <- rbind(tipdepth, tipdepthtemp)
		}
	times <- tipdepth
	}
	times
	}

#calculates the branching times of a tree with extant and extinct taxa
branching.times.complete <-function(tree){
    if (class(tree) != "phylo") 
        stop("object \"tree\" is not of class \"phylo\"")
    n <- length(tree$tip.label)
    N <- dim(tree$edge)[1]
    xx <- numeric(tree$Nnode)
    interns <- which(tree$edge[, 2] > n)
    for (i in interns) xx[tree$edge[i, 2] - n] <- xx[tree$edge[i, 
        1] - n] + tree$edge.length[i]
    depthtemp <- xx[tree$edge[, 1] - n] + tree$edge.length
    depth <- max(depthtemp)
    xx <- depth - xx
    names(xx) <- if (is.null(tree$node.label)) 
        (n + 1):(n + tree$Nnode)
    else tree$node.label
    xx
	}	

# ###END FROM TREESIM

# ###

cutvector <- function(datap){
	#temp <- 1:1000
#	maximum <- round(length(datap[,1])/100)
#	temp <-temp*min(maximum,length(datap[,1]))
#	newdatap<- datap[1,]
#	newdatap<-rbind(newdatap,datap[temp,])
#	newdatap<-rbind(newdatap,datap[length(datap[,1]),])
#	newdatap
	newdatap<-rbind(datap[1,],datap[2,1])
	iter <- round(length(datap[,1])/1000)
	temp<-3
	while (temp<= length(datap[,1])){
		if ((newdatap[length(newdatap[,1]),1] > -65.0001) && (newdatap[length(newdatap[,1]),1] < -64.9999) && (datap[temp,1] > -65.0001) && (datap[temp,1] < -64.9999)) {
			#print("-65")
			if (datap[temp,2] > newdatap[length(newdatap[,1]),2]) {
				newdatap[length(newdatap),2] <- datap[temp,2] 
			}
			temp<-temp+1
		} else {
			newdatap<-rbind(newdatap,datap[temp,])
			temp<-temp+iter
		}
	}
	newdatap
}
###

################################################################
# calculates branching times with number of lineages after that branching event
ltt.general <- function (t) {
branchingtree <- branching.times(t)
furcation <- table(t[[1]][,1])-1
furcation <- furcation[order(branchingtree)]
current <- furcation[length(furcation)]+1
linnumber <- current
for (j in 1:(length(furcation)-1)){
	current <- current + furcation[length(furcation)-j]
	linnumber <- c(linnumber, current)
	}
branchingtree <- -sort(branchingtree,decreasing=TRUE)
branchingtree <- c(branchingtree,0)
linnumber <- c(linnumber,linnumber[length(linnumber)])
obj<-cbind(branchingtree, linnumber)
obj
}


ltt.function <- function(time,t){
	ltt<-ltt.general(t)
	j=1
	while (ltt[j,1]<time) {
		j<-j+1
		}
	res <- ltt[j-1,2] + (ltt[j,2]-ltt[j-1,2])/(ltt[j,1]-ltt[j-1,1]) * (time - ltt[j-1,1])
	res
	}

# # Detect strange tips

count_dead <- function(phy){
  count <- length(getExtinct(phy, tol = 1e-8))
  return(count)
}

count_living <- function(phy){
  count <- length(getExtant(phy, tol = 1e-8))
  return(count)
}
