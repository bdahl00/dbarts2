.onUnload <- function(libpath)
{
  ## gc is necessary to collect external pointers who have not yet been collected
  ## that have finalizers pointing to the soon-to-unloaded dll
  gc(FALSE)
  if (is.loaded("dbarts_finalize", PACKAGE = "dbarts2")) {
    .Call(C_dbarts_finalize)
    library.dynam.unload("dbarts2", libpath)
  }
}
