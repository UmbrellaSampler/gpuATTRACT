      subroutine monte(cartstatehandle,ministatehandle,
     1 nhm, nihm, nlig,
     2 ens, phi, ssi, rot, xa, ya, za, morph, dlig,
     3 locrests, has_locrests,
     4 seed, label,
     5 gesa, energies, lablen)
c
c  variable metric minimizer (Harwell subroutine lib.  as in Jumna with modifications)
c     minimizes a single structure

      implicit none

c     Parameters
      integer cartstatehandle,ministatehandle
      include 'max.fin'
      integer nlig, seed
      real*8 locrests
      dimension locrests(3,maxlig)
      integer has_locrests
      dimension has_locrests(maxlig)
      real *8 gesa, energies, vbias
      dimension energies(6)
      integer lablen
      character label
      dimension label(lablen)

      integer nhm
      dimension nhm(maxlig)
      integer nihm
      dimension nihm(maxlig)
      integer ens, scaleens
      dimension ens(maxlig)
      real*8 phi, ssi, rot, dlig, xa, ya, za, morph
      dimension phi(maxlig), ssi(maxlig), rot(maxlig)
      dimension dlig(maxmode+maxindexmode, maxlig)
      dimension xa(maxlig), ya(maxlig), za(maxlig)
      dimension morph(maxlig)

ccc----------------ZHE--------------------
      integer mover, trials
<<<<<<< HEAD
      real*8 cumu_ws,sws
      dimension cumu_ws(maxmover)
=======
      real*8 cumu_sws,sws
      dimension cumu_sws(maxmover)
>>>>>>> e3da359249e1760bb15f6e8039cf95a766ca6393
ccc---------------------------------------

c     Local variables
      real*8 enew, energies0
      real*8 rrot1,rrot2,rrot3,rrot4,sphi,sssi,srot
      dimension energies0(6)
c     integer dseed,i,ii,j,jj,k,kk,itr,nfun
      integer i,ii,j,jj,k,kk,itr,nfun
      integer itra, ieig, iindex, iori, fixre, iscore,imcmax
      integer ju,ju0,jl,jb,nmodes,nimodes, jn, jn0
      integer iab,ijk,iaccept,accepts,inner_i
      real*8 xnull
      real*8 scalecenter,scalemode,ensprob,scalerot,rr
      real*8 rotmat,randrot,newrot,sum
      real*8 xaa,delta,deltamorph,bol,pi,mctemp,accept_rate
      integer ensaa
      real*8 dseed
      dimension xaa(maxdof)
      dimension ensaa(maxlig)
      dimension delta(maxdof), deltamorph(maxlig)
      dimension rr(maxdof),randrot(0:9),rotmat(0:9),newrot(0:9)
      integer nrens
      dimension nrens(maxlig)
      pointer(ptr_nrens,nrens)
      real*8 neomorph
      integer, parameter :: ERROR_UNIT = 0
      pi=3.141592654d0

cccc-------------------ZHE--------------------
C$$$  nmover = 3 [rigid_body_mover, ensemble_mover, both]
C$$$

      cumu_sws(1) = 5;
      cumu_sws(2) = 5;
      cumu_sws(3) = 10;
      sws = 0;
      do i=1,maxmover
         sws = sws+cumu_sws(i)
      enddo
      cumu_sws(1) = cumu_sws(1)/sws
      do i=2,maxmover
         cumu_sws(i) = cumu_sws(i-1)+cumu_sws(i)/sws
      enddo

      do i=1, maxlig
      ensaa(i) = 0
      enddo
c        call print_struc2(seed,label,gesa,energies,nlig,
c     1  ens,phi,ssi,rot,xa,ya,za,locrests,morph,
c     2  nhm,nihm,dlig,has_locrests,lablen)

      call ministate_f_monte(ministatehandle,
     1 iscore,imcmax,iori,itra,ieig,iindex,fixre,mctemp,
     2 scalerot,scalecenter,scalemode,ensprob)

c     always calculate only energies
      iab = 0

c
c  all variables without lig-hm
c
      jb=3*iori*(nlig-fixre)+3*itra*(nlig-fixre)
c  all variables including lig-hm
      nmodes = 0
      nimodes = 0
      do 5 i=fixre, nlig
      nmodes = nmodes + nhm(i)
      nimodes = nimodes + nihm(i)
    5 continue
      ju=jb+ieig*nmodes
      jn = ju + iindex*nimodes
      ju0=ju
      jn0 = jn
      do i=1,nlig
      if (morph(i).ge.0) then
      jn = jn + 1
      endif
      enddo

c  only trans or ori
      jl=3*iori*(nlig-fixre)

      call ministate_calc_pairlist(ministatehandle,cartstatehandle)
      call cartstate_get_nrens(cartstatehandle, ptr_nrens)

      xnull=0.0d0
      accept_rate=0.0d0
c     dseed=seed
      dseed=12345
      accepts=0
      scalerot=pi*scalerot/180d0
      nfun=0
      itr=0
      if (iscore.eq.1) then
        iori = 1
        itra = 1
      endif
c intial energy evaluation
c
      call energy(cartstatehandle,ministatehandle,
     1 iab,iori,itra,ieig,iindex,fixre,
     2 ens,phi,ssi,rot,xa,ya,za,morph,dlig,
     3 locrests, has_locrests, seed,
     4 gesa,energies,delta,deltamorph)
C$$$      if (iscore.eq.2) then
C$$$        call print_struc2(seed,label,gesa,energies,nlig,
C$$$     1  ens,phi,ssi,rot,xa,ya,za,locrests,morph,
C$$$     2  nhm,nihm,dlig,has_locrests,lablen)
C$$$      endif
c   start Monte Carlo
      trials=0
      iaccept=1
c      print*, "vmax: ", imcmax
      do 4000 ijk=1,imcmax

c         do 4100 inner_i=1, 1000
c      write (ERROR_UNIT,*), ijk, imcmax
c store old Euler angle, position and ligand and receptor coordinates
c
c phi,ssi,rot for first molecule are fixed!
      if(iaccept.eq.1) then
      do i=1,nlig
         ensaa(i)=ens(i)
      enddo
      if(iori.eq.1) then
      do 118 i=1+fixre,nlig
      ii=3*(i-fixre-1)
      xaa(ii+1)=phi(i)
      xaa(ii+2)=ssi(i)
      xaa(ii+3)=rot(i)
  118 continue
      endif
      if(itra.eq.1) then
      do 122 i=1+fixre,nlig
      ii=jl+3*(i-fixre-1)
      xaa(ii+1)=xa(i)
      xaa(ii+2)=ya(i)
      xaa(ii+3)=za(i)
  122 continue
      endif
      jj = jn0
      do i=1,nlig
       if (morph(i).ge.0) then
        xaa(jj+1) = morph(i)
        jj = jj + 1
       endif
      enddo

c if ligand flex is included store deformation factor in every mode in dlig(j)

      if(ieig.eq.1) then
      jj = 0
      do 130 j=1,nlig
      do 131 i=1,nhm(j)
      xaa(jb+jj+i)=dlig(i,j)
  131 continue
      jj = jj + nhm(j)
  130 continue
      endif

      if(iindex.eq.1) then
      jj = 0
      do 140 j=1,nlig
      do 141 i=1,nihm(j)
      xaa(ju0+jj+i)= dlig(ju0+i,j)
  141 continue
      jj = jj + nihm(j)
  140 continue
      endif

      endif
c old Cartesians are not stored!
c generate a total of ju random numbers
C$$$c      print*,"dseed 1",dseed
C$$$      call GGUBS(dseed,2,rr)
C$$$c      print*, "dseed 2 ",dseed

C$$$      call random_mover(rr(1),cumu_sws,mover,maxmover)
C$$$c      print*,"mover ",mover
C$$$      if ( mover.eq.1 ) trials = trials+1

C$$$        if (mover.eq.1 .or. mover.eq.3 .or. ensprob.eq.0) then
C$$$c           trials = trials+1
C$$$c           print*,"dseed 2p",dseed
C$$$           call rigid_body_mover(nlig,jl,iori,itra,phi,ssi,rot,xa,ya,za,
C$$$     1          fixre,scalerot,scalecenter,dseed)
C$$$c           print*, "dseed 3",dseed
C$$$        endif
C$$$        if (mover.eq.2 .or. mover.eq.3) then
C$$$           call GGUBS(dseed,3,rr)
C$$$c           print*, "dseed 4", dseed
C$$$           do i=1,nlig
C$$$              if (nrens(i).gt.0.and.morph(i).lt.0) then
C$$$C$$$                 call GGUBS(dseed,3,rr)
C$$$                 if (rr(1).lt.ensprob.and.rr(3).lt.float(i)/nlig) then
C$$$c	    ens(i) = int(rr(2)*nrens(i))+1
C$$$                    call enstrans(cartstatehandle,i-1,ens(i),rr(2),
C$$$     2                   ens(i))
C$$$                    exit
C$$$                 endif
C$$$              endif
C$$$           enddo
C$$$        endif


      call mc_ensemble_move(cartstatehandle,nlig,fixre,
     1 iori,itra,ens,nrens,ensprob,phi,ssi,rot,xa,ya,za,
     2 scalecenter,scalerot,cumu_sws,mover,dseed)

      if ( mover.eq.1 ) trials = trials+1

c make a move in HM direction and update x, y(1,i) and y(2,i) and dlig(j)
c     call crand(dseed,ju+1,rr)
      call GGUBS(dseed,jn+1,rr)
c     dseed = int(10000*rr(ju+1))
      if(ieig.eq.1) then
      kk = 0
      do 1180 k=1,nlig
      do 1200 i=1,nhm(k)
      dlig(i,k)=xaa(i+jb+kk)+scalemode*(rr(i+jb+kk)-0.5d0)
 1200 continue
      kk = kk + nhm(k)
 1180 continue
      endif
      if(iindex.eq.1) then
      kk = 0
      do 1280 k=1,nlig
      do 1300 i=1,nihm(k)
      dlig(ju0+i,k)=xaa(i+ju0+kk)+scalemode*(rr(i+ju0+kk)-0.5d0)
 1300 continue
      kk = kk+ nihm(k)
 1280 continue
      endif
c rigid body move, translation and rotation

c      call rigid_body_mover(nlig,jl,iori,itra,phi,ssi,rot,xa,ya,za,
c     1 fixre,scalerot,scalecenter,dseed)

      jj = jn0
      do i=1,nlig
       if (morph(i).ge.0) then
        neomorph = morph(i)+scalemode*(0.5d0-rr(ii+1))
        if (neomorph.lt.0) neomorph = 0
        if (neomorph.gt.nrens(i)-1.001) neomorph = nrens(i)-1.001
        morph(i) = neomorph
        jj = jj + 1
       endif
      enddo

      call energy(cartstatehandle,ministatehandle,
     1 iab,iori,itra,ieig,iindex,fixre,
     2 ens,phi,ssi,rot,xa,ya,za,morph,dlig,
     3 locrests, has_locrests, seed,
     4 enew,energies0,delta,deltamorph)
c  new energy
c      write (ERROR_UNIT,*),'Energy2', enew

c if using wte, bias energy
c      call evaluate_bias( enew, vbias);

      bol=enew-gesa
c      bol = enew + vbias - gesa
      if (mctemp.eq.0) then
      bol=sign(1.0d0,-bol)
      else
      bol=exp(-bol/mctemp)
      endif
c      write(*,*)'exp(bol)',enew,gesa,enew-gesa,bol
c     call crand(dseed,2,rr)
      call GGUBS(dseed,2,rr)
c     dseed = int(10000*rr(2))
      if(bol.gt.rr(1)) then
c      write(ERROR_UNIT,*)'accept the step', bol, rr(1)
c     write(*,*)
c    1 'rrot1,rrot2,rrot3,rrot4,sphi,phi(i),sssi,ssi(i),srot,rot(i)',
c    2 rrot1,rrot2,rrot3,rrot4,sphi,phi(2),sssi,ssi(2),srot,rot(2)
c      gesa=enew+vbias
c      call update_bias( enew );
        gesa=enew
        energies(:)=energies0(:)
      iaccept=1
<<<<<<< HEAD
      if (mover.eq.1 .or. mover.eq.3 .or. ensprob.eq.0) then
      	 accepts=accepts+1
      endif
=======
      if (mover.eq.1 ) accepts=accepts+1

>>>>>>> e3da359249e1760bb15f6e8039cf95a766ca6393
C$$$      print*,"iscore ",iscore
C$$$      if (iscore.eq.2) then
C$$$        call print_struc2(seed,label,gesa,energies,nlig,
C$$$     1  ens,phi,ssi,rot,xa,ya,za,locrests,morph,
C$$$     2  nhm,nihm,dlig,has_locrests,lablen)
C$$$      endif
c overwrite old xaa variables, see above
      else
c do not overwrite xaa variables
c      write(ERROR_UNIT,*)' step rejected'
      iaccept=0
      do i=1,nlig
         ens(i)=ensaa(i)
      enddo
      if(iori.eq.1) then
      do 1118 i=1+fixre,nlig
      ii=3*(i-fixre-1)
      phi(i)=xaa(ii+1)
      ssi(i)=xaa(ii+2)
      rot(i)=xaa(ii+3)
 1118 continue
      endif
      if(itra.eq.1) then
      do 1122 i=1+fixre,nlig
      ii=jl+3*(i-fixre-1)
      xa(i)=xaa(ii+1)
      ya(i)=xaa(ii+2)
      za(i)=xaa(ii+3)
 1122 continue
      endif

c if ligand flex is included store deformation factor in every mode in dlig(j)

      if(ieig.eq.1) then
      jj = 0
      do 230 j=1,nlig
      do 231 i=1,nhm(j)
      dlig(i,j)=xaa(jb+jj+i)
  231 continue
      jj = jj + nhm(j)
  230 continue
      endif

      if(iindex.eq.1) then
      jj = 0
      do 240 j=1,nlig
      do 241 i=1,nihm(j)
      dlig(ju0+i,j)=xaa(ju0+jj+i)
  241 continue
      jj = jj + nihm(j)
  240 continue
      endif
      endif

      jj = jn0
      do i=1,nlig
       if (morph(i).ge.0) then
        morph(i) = xaa(jj+1)
        jj = jj + 1
       endif
      enddo

c 4100 continue
<<<<<<< HEAD

c      if(mod(trials,50)==0) then
        if ( 1.eq.0 ) then
         accept_rate = real(accepts)/trials
         if(accept_rate.gt.0.5) then
            scalecenter=scalecenter*1.05
	    if ( scalerot*1.05<pi ) then
c control scalerot range in [0 pi]
               scalerot=scalerot*1.05
	    endif
         endif
         if(accept_rate.lt.0.5) then
            scalecenter=scalecenter*0.95
            scalerot=scalerot*0.95
         endif
      endif
      print*,"acc,tri,rot,center:",accepts,trials,scalerot,scalecenter
=======
c      print*,"accepts ", accepts

      if(mod(trials,25)==0) then
         accept_rate = real(accepts)/trials
         if(accept_rate.gt.0.3) then
            scalecenter=scalecenter*1.1
	    if ( scalerot*1.1<pi ) then
c control scalerot range in [0 pi]
               scalerot=scalerot*1.1
	    endif
         endif
         if(accept_rate.lt.0.3) then
            scalecenter=scalecenter*0.9
            scalerot=scalerot*0.9
         endif
      endif

>>>>>>> e3da359249e1760bb15f6e8039cf95a766ca6393

      if (iscore.eq.2) then
        call print_struc2(seed,label,gesa,energies,nlig,
     1  ens,phi,ssi,rot,xa,ya,za,locrests,morph,
     2  nhm,nihm,dlig,has_locrests,lablen)
      endif
 4000 continue
C$$$      print*,"2 phi ",phi
C$$$      print*,"2 ssi ",ssi
C$$$      print*,"2 rot ",rot
C$$$      print*,"2 xa ",xa
C$$$      print*,"2 ya ",yaS
C$$$      print*,"2 za ",za
c      print*,"scalerot scalecenter " , scalerot, scalecenter
c     Clean up
      call ministate_free_pairlist(ministatehandle)
      end
