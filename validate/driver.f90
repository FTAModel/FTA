! Runs GITM's FTA model (ModFtaModel.f90) at each AU, AL in points.txt and
! writes the results to f_<AU>_<-AL>.bin, for compare.py.
! Usage: ./driver <directory with the FTA coefficient files>/

program fta_driver

  use ModFTAModel
  implicit none

  real :: au, al
  integer :: iErr, ios
  character(len=200) :: fn

  call get_command_argument(1, dir)

  open(10, file='points.txt', status='old')
  do
    read(10, *, iostat=ios) au, al
    if (ios /= 0) exit
    call update_fta_model(au, al, iErr)
    if (iErr /= 0) stop 'Error in update_fta_model'
    write(fn, '(a,i0,a,i0,a)') 'f_', nint(au), '_', nint(-al), '.bin'
    open(11, file=trim(fn), access='stream', form='unformatted', status='replace')
    write(11) FtaAuVal, FtaAlVal, LBHLResult, LBHSResult, &
      eFluxResult, AveEResult, PolarCapResult
    close(11)
  enddo
  close(10)

end program fta_driver
