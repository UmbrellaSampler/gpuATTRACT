      subroutine mc_ensemble_move(cartstatehandle,nlig,fixre,
     1 iori,itra,ens,nrens,ensprob,phi,ssi,rot,xa,ya,za,
     2 scalecenter,scalerot,cumu_sws,mover,dseed)

      implicit none
      include 'max.fin'

C$$$      nlig: number of binding partners, it is 2 in 2-body docking
C$$$      fixre: if receptor is fixed
C$$$      itra: if translation is applied
C$$$      iori: if orientation is changed, or rotation is applied
C$$$      mover: applied mover index, needed later to count how many times rigid-body-mover applied
C$$$      mover: 1 - rigid-body-mover alone, 2 - enstrans only, 3 - rigid-body-mover and enstrans simutaneously
C$$$      ens: size maxlig, model index for each ligand(binding partner)
C$$$      nrens: size maxlig, model number for each binding partner
C$$$      scalecenter: magnitude for rigid-body translation
C$$$      scalerot: magnitude for rigid-body rotation
C$$$      cumu_sws: cumulative sampling weights for movers, options in the main code to define the sampling weights for each 3 movers
C$$$      dseed: random number seed, it is neccessary to maintail the dseed in the main code, otherwise it will be needed to initialize each time here


      integer cartstatehandle
      integer nlig,fixre,itra,iori
      integer i,mover,jl
      integer ens,nrens

      real*8 phi,ssi,rot,xa,ya,za,cumu_sws,dseed
      real*8 scalecenter,scalerot,rr,ensprob

      dimension ens(maxlig),nrens(maxlig)
      dimension phi(maxlig),ssi(maxlig),rot(maxlig)
      dimension xa(maxlig),ya(maxlig),za(maxlig)
      dimension cumu_sws(maxmover),rr(maxdof)

!      write(*,*) "ens:       ", ens
!      write(*,*) "nlig       ", nlig
!      write(*,*) "fixre      ", fixre
!      write(*,*) "iori       ", iori
!      write(*,*) "nrens      ", nrens
!      write(*,*) "ensprob    ", ensprob
!      write(*,*) "phi        ", phi
!      write(*,*) "ssi        ", ssi
!      write(*,*) "rot        ", rot
!      write(*,*) "xa         ", xa
!      write(*,*) "ya         ", ya
!      write(*,*) "za         ", za
!      write(*,*) "fixre      ", fixre
!      write(*,*) "scalerot   ", scalerot
!      write(*,*) "scalecenter", scalecenter
!      write(*,*) "cumu_sws   ", cumu_sws
!      write(*,*) "mover      ", mover
!      write(*,*) "dseed      ", dseed
!      write(*,*) ""

c      cartstatehandle,nlig,fixre,
c      iori,itra,ens,nrens,ensprob,phi,ssi,rot,xa,ya,za,
c      scalecenter,scalerot,cumu_sws,mover,dseed
C$$$      dseed=12345

      jl=3*iori*(nlig-fixre)
      call GGUBS(dseed,2,rr)

!      write(*,*) "rr         ", rr(1), rr(2)

      call random_mover(rr(1),cumu_sws,mover,maxmover)

!      write(*,*) "mover a. rm", mover
!      write(*,*) ""

      if (mover.eq.1 .or. mover.eq.3 .or. ensprob.eq.0) then
         call rigid_body_mover(nlig,jl,iori,itra,phi,ssi,rot,xa,ya,za,
     1        fixre,scalerot,scalecenter,dseed)
      endif
      if (mover.eq.2 .or. mover.eq.3) then
         call GGUBS(dseed,3,rr)
         do i=1,nlig
            if (nrens(i).gt.0) then

               if (rr(1).lt.ensprob.and.rr(3).lt.float(i)/nlig) then
                  call enstrans(cartstatehandle,i-1,ens(i),rr(2),
     2                 ens(i))
                  exit
               endif
            endif
         enddo
      endif

      end
