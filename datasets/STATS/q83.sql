select  count(*) from postLinks as pl,          posts as p,  		users as u,  		badges as b  where p.Id = pl.RelatedPostId  	and u.Id = p.OwnerUserId 	and u.Id = b.UserId  AND p.PostTypeId=1  AND p.Score<=12  AND p.AnswerCount=0  AND p.CommentCount=2  AND u.UpVotes>=0  AND u.CreationDate>='2011-01-24 21:09:11'::timestamp  AND u.CreationDate<='2014-09-12 22:21:49'::timestamp;