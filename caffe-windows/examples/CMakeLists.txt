# 把examples路径下的所有cpp文件的路径加入到examples_srcs中
file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/examples/*.cpp")

foreach(source_file ${examples_srcs})
  # get file name
  get_filename_component(name ${source_file} NAME_WE)

  # get folder name
  get_filename_component(path ${source_file} PATH)
  get_filename_component(folder ${path} NAME_WE)
  # 添加cpp文件到可执行程序
  add_executable(${name} ${source_file})
  # 链接动态链接库
  target_link_libraries(${name} ${Caffe_LINK})
  caffe_default_properties(${name})

  if(MSVC AND COPY_PREREQUISITES)
    caffe_copy_prerequisites(${name} USE_HARD_LINKS)
  endif()

  # set back RUNTIME_OUTPUT_DIRECTORY输出路径
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/examples/${folder}")

  caffe_set_solution_folder(${name} examples)

  # install
  install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})


  if(MSVC AND INSTALL_PREREQUISITES)
    caffe_install_prerequisites(${name} DESTINATION bin)
  endif()

  if(UNIX OR APPLE)
    # Funny command to make tutorials work
    # TODO: remove in future as soon as naming is standardized everywhere
    set(__outname ${PROJECT_BINARY_DIR}/examples/${folder}/${name}${Caffe_POSTFIX})
    add_custom_command(TARGET ${name} POST_BUILD
                       COMMAND ln -sf "${__outname}" "${__outname}.bin")
  endif()
endforeach()
