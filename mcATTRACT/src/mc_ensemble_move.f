      subroutine mc_ensemble_move(carstatehandle, nlig,
     1 ens, phi, ssi, rot, xa, ya, za, scalecenter, scalerot,
     2 ensprob, mover, cumu_sws)

      implicit none
      include 'max.fin'
      integer nlig, dseed
      real*8 phi, ssi, rot, xa, ya, za, cumu_sws
      real*8 scalecenter, scalerot, rr, ens
      dimension phi(maxlig), ssi(maxlig), rot(maxlig)
      dimension xa(maxlig), ya(maxlig), za(maxlig)
      dimension ens(maxlig)
      dimension cumu_sws(maxmover)
      dseed=12345

      call GGUBS(dseed,2,rr)

      call random_mover(rr(1),cumu_sws,mover,maxmover)
c      print*,"mover ",mover

        if (mover.eq.1 .or. mover.eq.3 .or. ensprob.eq.0) then
           trials = trials+1
           call rigid_body_mover(nlig,jl,iori,itra,phi,ssi,rot,xa,ya,za,
     1          fixre,scalerot,scalecenter,dseed)
        endif
        if (mover.eq.2 .or. mover.eq.3) then
           call GGUBS(dseed,3,rr)
           do i=1,nlig
              if (nrens(i).gt.0) then

                 if (rr(1).lt.ensprob.and.rr(3).lt.float(i)/nlig) then
c	    ens(i) = int(rr(2)*nrens(i))+1
                    call enstrans(cartstatehandle,i-1,ens(i),rr(2),
     2                   ens(i))
                    exit
                 endif
              endif
           enddo
        endif

      end


