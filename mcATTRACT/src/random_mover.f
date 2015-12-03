      subroutine random_mover(r,cumu_ws,mover)

      implicit none

c     Parameters
      include 'max.fin'
      integer mover,i
      real*8 cumu_ws,r
      dimension cumu_ws(maxmover)

      do i = 1,maxmover
        if ( cumu_ws(i)>=r ) then
           mover = i
           return
        endif
      enddo

      END
