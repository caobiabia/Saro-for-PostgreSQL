select  count(*) from comments as c,          postLinks as pl,          posts as p,  		users as u,  		badges as b   where p.Id = pl.RelatedPostId 	and p.Id = c.PostId 	and u.Id = b.UserId 	and u.Id = p.OwnerUserId  AND c.Score=0  AND c.CreationDate>='2010-07-30 17:40:15'::timestamp  AND pl.CreationDate>='2010-09-19 23:57:17'::timestamp  AND p.PostTypeId=2  AND p.Score=0  AND p.AnswerCount<=6  AND p.CreationDate<='2014-09-09 20:40:53'::timestamp  AND u.UpVotes<=26  AND u.CreationDate>='2011-06-14 14:04:30'::timestamp;