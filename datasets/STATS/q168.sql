select  count(*) from comments as c,          postLinks as pl,          posts as p,  		users as u,  		badges as b   where p.Id = pl.RelatedPostId 	and p.Id = c.PostId 	and u.Id = b.UserId 	and u.Id = p.OwnerUserId  AND p.Score<=18  AND p.AnswerCount>=0  AND p.CommentCount>=0  AND u.DownVotes>=0  AND u.DownVotes<=21  AND u.UpVotes>=0;