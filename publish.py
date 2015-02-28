#!env python
# script to sync publish this folder to a web server
import time

datetimeLog = {}
def loadTimestamp():
    # parse stored log
    import os
    timeFormat = '%Y-%m-%dT%H:%M:%SZ'
    logFilename = '.log'
    if os.path.exists(logFilename):
        f = open(logFilename, 'r')
        lines = f.readlines()
        for line in lines:
            [a, b] = line.strip().split(' ')
            datetimeLog[a] = time.strptime(b, timeFormat)

def writeTimestamp():
    # save updated log
    logFilename = '.log'
    timeFormat = '%Y-%m-%dT%H:%M:%SZ'
    f = open(logFilename, 'w')
    for filename in datetimeLog.keys():
        f.write(filename + ' ' + time.strftime(timeFormat, datetimeLog[filename]) + '\n')
    f.close()

def s3_update():
    ''' Script to upload file to s3 '''
    import boto
    import boto.s3
     
    import os.path
    import sys
     
    # Fill in info on data to upload
    # destination bucket name
    bucket_name = 'arbib-chapter-demos'
    # source directory
    sourceDir = '.'
    # destination directory name (on s3)
    destDir = ''
     
    #max size in bytes before uploading in parts. between 1 and 5 GB recommended
    MAX_SIZE = 20 * 1000 * 1000
    #size of parts when uploading in parts
    PART_SIZE = 6 * 1000 * 1000
     
    #conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_ACCESS_KEY_SECRET)
    conn = boto.connect_s3(host='s3-us-west-2.amazonaws.com') # a bug of boto requires to set the host, https://github.com/boto/boto/issues/621
     
    #bucket = conn.create_bucket(bucket_name,
            #location=boto.s3.connection.Location.DEFAULT)
    bucket = conn.get_bucket(bucket_name)
     
    uploadFileNames = []
    for (root, dirs, files) in os.walk(sourceDir):
        #print filename
        filenames = [root + '/' + item for item in files]
        #uploadFileNames.extend(filename)
        uploadFileNames.extend(filenames)
        #break

    # exclude image file
    def exclude(filename):
        dirs = filename.split('/')[:-1]
        excludeFolder = ['.git', 'trash', 'reference', '.ipynb_checkpoints']
        for folder in dirs:
            if folder in excludeFolder:
                return True

        for ext in ['.py', '.git', '.swp', '.zip', '.DS_Store', '.log', '.gitignore']:
            if filename.endswith(ext):
                return True
        else:
            return False

    uploadFileNames = [a for a in uploadFileNames if not exclude(a)]
     
    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()


     
    for filename in uploadFileNames:
        updateTime = time.strptime(time.ctime(os.path.getmtime(filename)))
        if datetimeLog.get(filename) and updateTime <= datetimeLog[filename]:
            continue

        #sourcepath = os.path.join(sourceDir + filename)
        sourcepath = filename
        destpath = os.path.join(destDir, filename[len(sourceDir)+1:])
        print 'Uploading %s --> %s to Amazon S3 bucket %s' % \
               (sourcepath, destpath, bucket_name)
     
        filesize = os.path.getsize(sourcepath)
        if filesize > MAX_SIZE:
            print "multipart upload"
            mp = bucket.initiate_multipart_upload(destpath)
            fp = open(sourcepath,'rb')
            fp_num = 0
            while (fp.tell() < filesize):
                fp_num += 1
                print "uploading part %i" %fp_num
                mp.upload_part_from_file(fp, fp_num, cb=percent_cb, num_cb=10, size=PART_SIZE)
     
            mp.complete_upload()
     
        else:
            print "singlepart upload"
            k = boto.s3.key.Key(bucket)
            k.key = destpath
            k.set_contents_from_filename(sourcepath,
                    cb=percent_cb, num_cb=10)
        
        datetimeLog[filename] = updateTime



def exportHTML():
    import glob
    import os
    import IPython.nbconvert
    from IPython.nbformat import current as nbformat

    files = glob.glob('./ipynb/*.ipynb')
    HTMLDir = './html/'
    for filename in files:
        updateTime = time.strptime(time.ctime(os.path.getmtime(filename)))
        if datetimeLog.get(filename) and updateTime <= datetimeLog[filename]:
            continue

        with open(filename, 'r') as f:
            notebook = nbformat.read(f, 'ipynb')

        from IPython.nbconvert import HTMLExporter
        from IPython.config import Config
        exportHtml = HTMLExporter(config=None, template_file='./data/other/full.tpl')
        
        htmlFilename = HTMLDir + filename.split('/')[-1].replace('.ipynb', '.html')
        ipynbFilename = '../ipynb/' + filename.split('/')[-1]
        print 'Convert from %s to %s' % (ipynbFilename, htmlFilename)
        (body,resources) = exportHtml.from_notebook_node(notebook, resources={'filename':ipynbFilename})
        with open(htmlFilename, 'w') as f:
            f.write(body.encode('utf-8'))

        datetimeLog[filename] = updateTime

loadTimestamp()           
exportHTML()        
s3_update()
writeTimestamp()